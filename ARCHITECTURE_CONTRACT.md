# Final MPD-Transformer architecture contract

## Scientific role

The model is an angle3-conditioned bilayer tactile source separator. Its
Transformer operation models spatial interaction between Map1 and Map2; it
does not model raw6 as an IMU time sequence.

## Exact inputs

- `Map1`: normalized tensor `[B, 3, 32, 32]`
- `Map2`: normalized tensor `[B, 3, 32, 32]`
- `angle3`: normalized tensor `[B, 3]`

`angle3` is supplied by the independently estimated orientation pathway. The
canonical dataset stores the aligned angle3 values directly. `raw6` is retained
in the standalone data copy for provenance, but is not accepted by the model
API.

## Exact computation

1. Two layer-specific shallow tactile encoders map each dense vector field to
   1,024 spatial tokens of width 128.
2. Both token streams receive 2D sinusoidal positional encoding.
3. A shared angle3 embedding produces early, branch-specific FiLM-like
   modulation of both tactile streams.
4. Synchronous bidirectional cross-attention computes Map1-from-Map2 and
   Map2-from-Map1 interactions with four heads.
5. Each direction uses a content-adaptive sigmoid gate followed by an
   independent `128 -> 512 -> 128` feed-forward update.
6. Both updated bilayer streams enter each of two distinct semantic source
   decoders:
   - External decoder
   - Internal decoder
7. Each source decoder produces a Map1 field and a Map2 field.

## Exact outputs

- `ext_map1`: `[B, 3, 32, 32]`
- `ext_map2`: `[B, 3, 32, 32]`
- `int_map1`: `[B, 3, 32, 32]`
- `int_map2`: `[B, 3, 32, 32]`

The implementation has 1,217,234 trainable parameters. It contains no pose
head, coupling-field head, raw6 branch, or fifth semantic output. A
reconstruction remainder may be calculated after inference as a diagnostic
only.
