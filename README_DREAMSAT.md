# DreamSat geometry-only path

This fork adds `Trellis2GeometryOnlyPipeline`, which loads only the sparse-structure and shape models. It deliberately excludes:

- `tex_slat_flow_model_512`
- `tex_slat_flow_model_1024`
- `tex_slat_decoder`
- texture SLat sampling
- PBR voxel attributes and texture baking

## Run

```bash
python generate_geometry_only.py \
  --image assets/example_image/T.png \
  --output sample_geometry.obj \
  --pipeline-type 1024_cascade
```

The output is a geometry-only OBJ/PLY/GLB supported by `trimesh`, containing vertices and faces without generated materials.

## Why this saves memory

`Pipeline.from_pretrained()` filters model entries using `model_names_to_load`. The geometry-only pipeline contains only:

```text
sparse_structure_flow_model
sparse_structure_decoder
shape_slat_flow_model_512
shape_slat_flow_model_1024
shape_slat_decoder
```

Therefore the texture DiTs and texture VAE are not instantiated. The inference path ends after `decode_shape_slat()` and returns `Mesh` objects.

## Runtime measurements

`generate_geometry_only.py` prints peak CUDA allocation and reservation. Compare these values against the original `example.py` using the same GPU, resolution, seed, and input image.
