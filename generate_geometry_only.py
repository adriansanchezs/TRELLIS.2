#!/usr/bin/env python3
"""DreamSat geometry-only inference and VRAM measurement."""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import trimesh
from PIL import Image

from trellis2.pipelines import Trellis2GeometryOnlyPipeline


def export_mesh(mesh, path: str) -> None:
    vertices = mesh.vertices.detach().cpu().numpy()
    faces = mesh.faces.detach().cpu().numpy()
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("The generated mesh has no vertices or faces.")
    output = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    output.export(path)
    print(f"Exported: {path}")
    print(f"Vertices: {len(output.vertices)}")
    print(f"Faces: {len(output.faces)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="assets/example_image/T.png")
    parser.add_argument("--output", default="sample_geometry.obj")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--pipeline-type", choices=["512", "1024", "1024_cascade", "1536_cascade"], default="1024_cascade")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DreamSat geometry-only inference requires CUDA.")

    pipeline = Trellis2GeometryOnlyPipeline.from_pretrained(args.model)
    pipeline.cuda()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()

    image = Image.open(args.image)
    start = time.perf_counter()
    meshes = pipeline.run(
        image,
        seed=args.seed,
        pipeline_type=args.pipeline_type,
        max_num_tokens=args.max_num_tokens,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    export_mesh(meshes[0], args.output)
    print(f"Inference seconds: {elapsed:.3f}")
    print(f"Allocated before: {allocated_before / 2**30:.3f} GiB")
    print(f"Reserved before: {reserved_before / 2**30:.3f} GiB")
    print(f"Peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.3f} GiB")
    print(f"Peak reserved: {torch.cuda.max_memory_reserved() / 2**30:.3f} GiB")


if __name__ == "__main__":
    main()
