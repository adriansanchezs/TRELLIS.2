from typing import *

import torch

from .trellis2_image_to_3d import Trellis2ImageTo3DPipeline
from . import samplers, rembg
from ..modules import image_feature_extractor


class Trellis2GeometryOnlyPipeline(Trellis2ImageTo3DPipeline):
    """TRELLIS.2 inference path that generates geometry only.

    Texture/Material SLat flow models and the texture decoder are deliberately
    excluded from ``model_names_to_load`` and are never sampled or decoded.
    The returned objects are ``Mesh`` instances containing vertices and faces.
    """

    model_names_to_load = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    ]

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
    ) -> "Trellis2GeometryOnlyPipeline":
        # Pipeline.from_pretrained() honors model_names_to_load, so the three
        # texture checkpoints are skipped before they are instantiated.
        pipeline = super(Trellis2ImageTo3DPipeline, cls).from_pretrained(
            path,
            config_file,
        )
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(
            samplers,
            args["sparse_structure_sampler"]["name"],
        )(**args["sparse_structure_sampler"]["args"])
        pipeline.sparse_structure_sampler_params = (
            args["sparse_structure_sampler"]["params"]
        )

        pipeline.shape_slat_sampler = getattr(
            samplers,
            args["shape_slat_sampler"]["name"],
        )(**args["shape_slat_sampler"]["args"])
        pipeline.shape_slat_sampler_params = (
            args["shape_slat_sampler"]["params"]
        )

        pipeline.shape_slat_normalization = args["shape_slat_normalization"]
        pipeline.image_cond_model = getattr(
            image_feature_extractor,
            args["image_cond_model"]["name"],
        )(**args["image_cond_model"]["args"])
        pipeline.rembg_model = getattr(
            rembg,
            args["rembg_model"]["name"],
        )(**args["rembg_model"]["args"])
        pipeline.low_vram = args.get("low_vram", True)
        pipeline.default_pipeline_type = args.get(
            "default_pipeline_type",
            "1024_cascade",
        )
        pipeline._device = "cpu"
        return pipeline

    @torch.no_grad()
    def decode_geometry(
        self,
        shape_slat,
        resolution: int,
    ):
        """Decode Shape SLat and discard texture-only substructures."""
        meshes, _ = self.decode_shape_slat(shape_slat, resolution)
        for mesh in meshes:
            mesh.fill_holes()
        return meshes

    @torch.no_grad()
    def run(
        self,
        image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ):
        """Generate one or more clean geometric meshes."""
        sparse_structure_sampler_params = sparse_structure_sampler_params or {}
        shape_slat_sampler_params = shape_slat_sampler_params or {}
        pipeline_type = pipeline_type or self.default_pipeline_type

        if pipeline_type not in {
            "512",
            "1024",
            "1024_cascade",
            "1536_cascade",
        }:
            raise ValueError(f"Invalid pipeline type: {pipeline_type}")

        if preprocess_image:
            image = self.preprocess_image(image)

        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = (
            self.get_cond([image], 1024)
            if pipeline_type != "512"
            else None
        )
        structure_resolution = {
            "512": 32,
            "1024": 64,
            "1024_cascade": 32,
            "1536_cascade": 32,
        }[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512,
            structure_resolution,
            num_samples,
            sparse_structure_sampler_params,
        )

        if pipeline_type == "512":
            shape_slat = self.sample_shape_slat(
                cond_512,
                self.models["shape_slat_flow_model_512"],
                coords,
                shape_slat_sampler_params,
            )
            resolution = 512
        elif pipeline_type == "1024":
            shape_slat = self.sample_shape_slat(
                cond_1024,
                self.models["shape_slat_flow_model_1024"],
                coords,
                shape_slat_sampler_params,
            )
            resolution = 1024
        else:
            target_resolution = 1024 if pipeline_type == "1024_cascade" else 1536
            shape_slat, resolution = self.sample_shape_slat_cascade(
                cond_512,
                cond_1024,
                self.models["shape_slat_flow_model_512"],
                self.models["shape_slat_flow_model_1024"],
                512,
                target_resolution,
                coords,
                shape_slat_sampler_params,
                max_num_tokens,
            )

        meshes = self.decode_geometry(shape_slat, resolution)
        torch.cuda.empty_cache()
        if return_latent:
            return meshes, (shape_slat, resolution)
        return meshes
