from __future__ import annotations

import json
import math
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, BinaryIO

import psutil

from src.final_validation import (
    DISCONTINUED_MARKER,
    validate_finalization,
    validate_oracle_completion,
    validate_seed22_discontinuation,
)
from src.utils import atomic_write_json, read_json, sha256_file


ROOT = Path(__file__).resolve().parent
LOG_ROOT = ROOT / "logs"
RUNS_ROOT = ROOT / "runs"
FINAL_ROOT = ROOT / "final"
CONFIGS = (
    ROOT / "configs" / "candidate_seed11.json",
    ROOT / "configs" / "candidate_seed22.json",
)
MAX_RESTARTS = 10
WORKER_STALE_SECONDS = 20 * 60


def current_source_hashes() -> dict[str, str]:
    return {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in (
            ROOT / "train.py",
            ROOT / "src" / "model.py",
            ROOT / "src" / "data.py",
            ROOT / "src" / "metrics.py",
            ROOT / "src" / "utils.py",
        )
    }


def candidate_completion_check(
    config_path: Path,
    config: dict[str, Any],
) -> tuple[bool, str | None]:
    run_root = RUNS_ROOT / config["run_id"]
    marker_path = run_root / "CANDIDATE_COMPLETE.json"
    result_path = run_root / "candidate_result.json"
    if not marker_path.is_file() and not result_path.is_file():
        return False, None
    if not marker_path.is_file() or not result_path.is_file():
        existing = marker_path if marker_path.is_file() else result_path
        if time.time() - existing.stat().st_mtime < 120:
            return False, None
        return False, "completion marker/result presence is inconsistent"
    try:
        marker = read_json(marker_path)
        result = read_json(result_path)
        checkpoint = ROOT / str(result["best_checkpoint"])
        if marker.get("status") != "candidate_complete":
            raise RuntimeError("marker status")
        if result.get("status") != "candidate_complete":
            raise RuntimeError("result status")
        if result.get("run_id") != config["run_id"]:
            raise RuntimeError("run id")
        if marker.get("result_sha256") != sha256_file(result_path):
            raise RuntimeError("result hash")
        if result.get("config_sha256") != sha256_file(config_path):
            raise RuntimeError("config hash")
        if result.get("data_manifest_sha256") != sha256_file(
            ROOT / "data" / "data_manifest.json"
        ):
            raise RuntimeError("data manifest hash")
        if result.get("source_sha256") != current_source_hashes():
            raise RuntimeError("source hashes")
        if not checkpoint.is_file():
            raise RuntimeError("best checkpoint missing")
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != result.get("best_checkpoint_sha256"):
            raise RuntimeError("checkpoint result hash")
        if checkpoint_hash != marker.get("best_checkpoint_sha256"):
            raise RuntimeError("checkpoint marker hash")
        if not math.isfinite(
            float(result.get("best_mean_field_nrmse", math.nan))
        ):
            raise RuntimeError("non-finite best metric")
        return True, None
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        return False, repr(error)


def archive_invalid_completion(config: dict[str, Any], reason: str) -> None:
    run_root = (RUNS_ROOT / config["run_id"]).resolve()
    run_root.relative_to(RUNS_ROOT.resolve())
    archive_root = (
        run_root
        / "failures"
        / (
            f"invalid_completion_{time.strftime('%Y%m%dT%H%M%S')}_"
            f"{os.getpid()}_{time.time_ns()}"
        )
    )
    archive_root.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for name in ("CANDIDATE_COMPLETE.json", "candidate_result.json"):
        source = run_root / name
        if source.is_file():
            destination = archive_root / name
            os.replace(source, destination)
            moved.append(name)
    atomic_write_json(
        archive_root / "ARCHIVED.json",
        {
            "status": "invalid_completion_archived_for_resume",
            "run_id": config["run_id"],
            "reason": reason,
            "moved_files": moved,
            "archived_unix": time.time(),
        },
    )


def acquire_lock() -> BinaryIO:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / "supervisor.lock"
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        handle.close()
        raise RuntimeError("Another New_training supervisor owns the lock") from error
    return handle


def candidate_state(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    run_root = RUNS_ROOT / config["run_id"]
    status_path = run_root / "status.json"
    complete, completion_error = candidate_completion_check(config_path, config)
    state: dict[str, Any] = {
        "run_id": config["run_id"],
        "complete": complete,
    }
    if completion_error is not None:
        state["completion_error"] = completion_error
    if status_path.is_file():
        try:
            state["status"] = read_json(status_path)
        except Exception as error:
            state["status_read_error"] = repr(error)
    return state


def independently_running_worker(config: dict[str, Any]) -> dict[str, Any] | None:
    """Recognize a trainer that survived a supervisor restart."""

    owner_path = RUNS_ROOT / config["run_id"] / "worker_owner.json"
    if not owner_path.is_file():
        return None
    try:
        owner = read_json(owner_path)
        process = psutil.Process(int(owner["pid"]))
        if abs(process.create_time() - float(owner["create_time"])) > 1e-3:
            return None
        command = " ".join(process.cmdline()).lower()
        expected_config = str(owner.get("config", "")).lower()
        if (
            "train.py" not in command
            or expected_config not in command
            or owner.get("run_id") != config["run_id"]
        ):
            return None
        activity_values = [float(owner.get("updated_unix", 0.0))]
        status_path = RUNS_ROOT / config["run_id"] / "status.json"
        if status_path.is_file():
            try:
                status = read_json(status_path)
                activity_values.append(float(status.get("updated_unix", 0.0)))
            except (TypeError, ValueError, OSError, json.JSONDecodeError):
                activity_values.append(status_path.stat().st_mtime)
        last_activity = max(activity_values)
        return {
            "pid": process.pid,
            "create_time": process.create_time(),
            "status": process.status(),
            "adopted": True,
            "last_activity_unix": last_activity,
            "stale": time.time() - last_activity > WORKER_STALE_SECONDS,
        }
    except (KeyError, ValueError, psutil.Error, OSError):
        return None


def terminate_worker(pid: int, run_id: str, reason: str) -> None:
    try:
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        for child in reversed(children):
            child.terminate()
        process.terminate()
        _, alive = psutil.wait_procs(children + [process], timeout=30)
        for survivor in alive:
            survivor.kill()
        psutil.wait_procs(alive, timeout=15)
    except psutil.NoSuchProcess:
        pass
    atomic_write_json(
        RUNS_ROOT
        / run_id
        / "failures"
        / f"supervisor_termination_{time.strftime('%Y%m%dT%H%M%S')}.json",
        {
            "status": "terminated_for_recovery",
            "run_id": run_id,
            "pid": pid,
            "reason": reason,
            "terminated_unix": time.time(),
        },
    )


def launch_candidate(
    config_path: Path,
    config: dict[str, Any],
) -> tuple[subprocess.Popen[bytes], BinaryIO]:
    run_root = RUNS_ROOT / config["run_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "process.log"
    log_handle = log_path.open("ab", buffering=0)
    command = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(config_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    return process, log_handle


def run_checked(command: list[str], log_name: str, attempts: int = 3) -> None:
    log_path = LOG_ROOT / log_name
    for attempt in range(1, attempts + 1):
        with log_path.open("ab", buffering=0) as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check=False,
            )
        if completed.returncode == 0:
            return
        if attempt < attempts:
            time.sleep(30)
    raise RuntimeError(f"Command failed after {attempts} attempts: {command}")


def main() -> int:
    lock_handle = acquire_lock()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    complete_path = ROOT / "ORACLE_COMPLETE.json"
    if complete_path.is_file():
        validation = validate_oracle_completion(ROOT)
        if validation["passed"]:
            return 0
        raise RuntimeError(
            f"ORACLE_COMPLETE contract failed: {validation['errors']}"
        )

    configs = [(path, read_json(path)) for path in CONFIGS]
    discontinuation = validate_seed22_discontinuation(ROOT)
    if not discontinuation["passed"]:
        raise RuntimeError(
            "Seed 22 discontinuation evidence is invalid: "
            f"{discontinuation['errors']}"
        )
    planned_configs = configs
    configs = [planned_configs[0]]
    count_contract = {
        "candidate_mode": "seed11-only-after-discontinuation",
        "planned_candidate_count": 2,
        "completed_candidate_count": 1,
        "selection_eligible_candidate_count": 1,
        "discontinued_incomplete_candidate_count": 1,
        "discontinued_candidate_marker": DISCONTINUED_MARKER,
        "discontinued_candidate_marker_sha256": discontinuation[
            "marker_sha256"
        ],
    }
    processes: dict[str, tuple[subprocess.Popen[bytes], BinaryIO]] = {}
    restarts = {config["run_id"]: 0 for _, config in configs}
    atomic_write_json(
        LOG_ROOT / "supervisor_owner.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "python": sys.executable,
            "started_unix": time.time(),
            **count_contract,
        },
    )
    try:
        while True:
            all_complete = True
            for config_path, config in configs:
                run_id = config["run_id"]
                run_root = RUNS_ROOT / run_id
                candidate_complete, completion_error = candidate_completion_check(
                    config_path, config
                )
                if completion_error is not None:
                    archive_invalid_completion(config, completion_error)
                    candidate_complete = False
                if candidate_complete:
                    if run_id in processes:
                        process, log_handle = processes.pop(run_id)
                        if process.poll() is None:
                            process.wait(timeout=60)
                        log_handle.close()
                    continue
                all_complete = False
                if run_id not in processes:
                    adopted = independently_running_worker(config)
                    if adopted is not None:
                        if not adopted["stale"]:
                            continue
                        terminate_worker(
                            int(adopted["pid"]),
                            run_id,
                            (
                                "No status/checkpoint progress for "
                                f"{WORKER_STALE_SECONDS} seconds"
                            ),
                        )
                        time.sleep(5)
                    process, log_handle = launch_candidate(config_path, config)
                    processes[run_id] = (process, log_handle)
                    restarts[run_id] += 1
                else:
                    process, log_handle = processes[run_id]
                    returncode = process.poll()
                    active = independently_running_worker(config)
                    if (
                        returncode is None
                        and active is not None
                        and active["stale"]
                    ):
                        terminate_worker(
                            process.pid,
                            run_id,
                            (
                                "No status/checkpoint progress for "
                                f"{WORKER_STALE_SECONDS} seconds"
                            ),
                        )
                        returncode = process.wait(timeout=45)
                    elif returncode is None and active is None:
                        try:
                            launched = psutil.Process(process.pid).create_time()
                        except psutil.Error:
                            launched = time.time()
                        if time.time() - launched > WORKER_STALE_SECONDS:
                            terminate_worker(
                                process.pid,
                                run_id,
                                (
                                    "Worker never published valid owner/status "
                                    f"within {WORKER_STALE_SECONDS} seconds"
                                ),
                            )
                            returncode = process.wait(timeout=45)
                    if returncode is not None:
                        log_handle.close()
                        del processes[run_id]
                        if restarts[run_id] >= MAX_RESTARTS:
                            raise RuntimeError(
                                f"{run_id} exhausted {MAX_RESTARTS} restarts; "
                                f"last return code {returncode}"
                            )
                        time.sleep(15)
                        process, new_log_handle = launch_candidate(config_path, config)
                        processes[run_id] = (process, new_log_handle)
                        restarts[run_id] += 1
            status = {
                "status": "training" if not all_complete else "finalizing",
                "progress": (
                    f"{sum(candidate_state(p, c)['complete'] for p, c in configs)}/"
                    f"{len(configs)} selection-eligible candidates complete; "
                    "1/2 planned candidates complete and 1 discontinued"
                ),
                "candidates": [
                    candidate_state(path, config) for path, config in configs
                ]
                + [
                    {
                        "run_id": discontinuation["marker"]["run_id"],
                        "complete": False,
                        "selection_eligible": False,
                        "status": discontinuation["marker"]["status"],
                        "observed_terminal_epoch": discontinuation["marker"][
                            "observed_terminal_epoch"
                        ],
                        "planned_target_epoch": discontinuation["marker"][
                            "planned_target_epoch"
                        ],
                    }
                ],
                "processes": {
                    run_id: {
                        "pid": process.pid,
                        "returncode": process.poll(),
                        "restart_attempt": restarts[run_id],
                    }
                    for run_id, (process, _) in processes.items()
                },
                "adopted_workers": {
                    config["run_id"]: adopted
                    for _, config in configs
                    if (
                        (adopted := independently_running_worker(config))
                        is not None
                        and config["run_id"] not in processes
                    )
                },
                "updated_unix": time.time(),
            }
            atomic_write_json(LOG_ROOT / "supervisor_status.json", status)
            if all_complete:
                break
            time.sleep(30)

        run_checked(
            [
                sys.executable,
                str(ROOT / "finalize.py"),
                "--device",
                "cuda:0",
                "--candidate-mode",
                "seed11-only-after-discontinuation",
            ],
            "finalize.log",
        )
        run_checked(
            [sys.executable, str(ROOT / "verify_final.py"), "--device", "cuda:0"],
            "verify_final.log",
        )
        verification = FINAL_ROOT / "FINAL_VERIFICATION.json"
        selection = FINAL_ROOT / "selected_model.json"
        final_validation = validate_finalization(
            ROOT, require_verification=True
        )
        if not final_validation["passed"]:
            raise RuntimeError(
                f"Final assets are invalid: {final_validation['errors']}"
            )
        marker = {
            "schema_version": 1,
            "status": "oracle_complete",
            **count_contract,
            "selected_model": selection.relative_to(ROOT).as_posix(),
            "selected_model_sha256": sha256_file(selection),
            "final_verification": verification.relative_to(ROOT).as_posix(),
            "final_verification_sha256": sha256_file(verification),
            "model_bundle": "final/model_bundle.pt",
            "model_bundle_sha256": sha256_file(FINAL_ROOT / "model_bundle.pt"),
            "finalization_sha256": sha256_file(
                FINAL_ROOT / "FINALIZATION_COMPLETE.json"
            ),
            "final_metrics_sha256": sha256_file(
                FINAL_ROOT / "final_metrics.json"
            ),
            "source_manifest_sha256": sha256_file(
                FINAL_ROOT / "source_manifest.json"
            ),
            "completed_unix": time.time(),
        }
        atomic_write_json(complete_path, marker)
        oracle_validation = validate_oracle_completion(ROOT)
        if not oracle_validation["passed"]:
            raise RuntimeError(
                f"Published ORACLE_COMPLETE failed validation: "
                f"{oracle_validation['errors']}"
            )
        atomic_write_json(
            LOG_ROOT / "supervisor_status.json",
            {
                "status": "complete",
                "progress": (
                    "1/1 selection-eligible candidate complete; "
                    "Seed 22 discontinued incomplete; final verification complete"
                ),
                **count_contract,
                "updated_unix": time.time(),
            },
        )
        return 0
    except BaseException as error:
        atomic_write_json(
            LOG_ROOT / f"supervisor_failure_{time.strftime('%Y%m%dT%H%M%S')}.json",
            {
                "status": "failed",
                "pid": os.getpid(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "failed_unix": time.time(),
            },
        )
        raise
    finally:
        for process, log_handle in processes.values():
            if process.poll() is not None:
                log_handle.close()
        lock_handle.seek(0)
        msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
