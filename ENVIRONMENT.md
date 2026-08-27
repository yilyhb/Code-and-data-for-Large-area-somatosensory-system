# Environment

The finalized and verified Windows environment was:

- Python 3.10.16
- PyTorch 2.4.1
- CUDA 12.4
- cuDNN 9.1
- NumPy 1.26.4
- Matplotlib 3.9.2
- two NVIDIA GeForce RTX 4090 GPUs

For the closest GPU reproduction:

```powershell
conda env create -f environment.yml
conda activate mpd-repro-py310
```

The inference example also runs on CPU:

```powershell
python use\inference_visualization_example.py --index 3500 --device cpu
```

`requirements.txt` records the Python package versions, but the Conda
environment is preferred because it also specifies the CUDA runtime. The
original Windows x86-64 explicit Conda lock is retained as
`environment-explicit-windows.txt`.

