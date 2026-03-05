# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Model cache for ViPE - loads expensive models once and reuses them across videos.

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import DictConfig

from vipe.priors.depth import make_depth_model
from vipe.priors.depth.adapter import PinholeDepthAdapter
from vipe.priors.depth.base import DepthType
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.networks.droid_net import DroidNet
from vipe.utils.cameras import CameraType


logger = logging.getLogger(__name__)


@dataclass
class VipeModelCache:
    """
    Cache for expensive ViPE models that should be loaded once and reused.
    
    Usage:
        cache = VipeModelCache.create(slam_cfg, init_cfg, post_cfg, device)
        # Now pass cache to SLAMSystem and processors
    """
    
    # Core SLAM model
    droid_net: Optional[DroidNet] = None
    
    # Metric depth model (e.g., UniDepth)
    metric_depth: Optional[object] = None
    metric_depth_name: Optional[str] = None
    
    # GeoCalib for intrinsics estimation
    geocalib_pinhole: Optional[GeoCalib] = None
    geocalib_distorted: Optional[GeoCalib] = None
    
    # Adaptive depth models
    video_depth_model: Optional[VideoDepthAnythingDepthModel] = None
    video_depth_model_name: Optional[str] = None
    prior_da_model: Optional[PriorDAModel] = None
    adaptive_metric_depth: Optional[object] = None
    adaptive_metric_depth_name: Optional[str] = None
    
    # TrackAnything (SAM + AOT) for instance segmentation
    track_anything: Optional[TrackAnythingPipeline] = None
    
    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))
    
    @classmethod
    def create(
        cls,
        slam_cfg: Optional[DictConfig] = None,
        init_cfg: Optional[DictConfig] = None,
        post_cfg: Optional[DictConfig] = None,
        device: torch.device = torch.device("cuda"),
    ) -> "VipeModelCache":
        """
        Create a model cache by loading all required models.
        
        Args:
            slam_cfg: SLAM config (for metric depth model)
            init_cfg: Init config (for GeoCalib)
            post_cfg: Post config (for adaptive depth)
            device: CUDA device
        """
        cache = cls(device=device)
        
        # Load DroidNet (always needed)
        logger.info("Loading DroidNet...")
        cache.droid_net = DroidNet().to(device)
        
        # Load metric depth for SLAM if configured
        if slam_cfg is not None and slam_cfg.get("keyframe_depth") is not None:
            depth_name = slam_cfg.keyframe_depth
            logger.info(f"Loading SLAM metric depth model: {depth_name}")
            cache.metric_depth = make_depth_model(depth_name)
            cache.metric_depth_name = depth_name
            # Will be adapted to pinhole if needed when used
        
        # Load GeoCalib models
        if init_cfg is not None:
            camera_type = CameraType(init_cfg.get("camera_type", "pinhole"))
            fixed_fov = init_cfg.get("fixed_fov_degrees", None)
            
            if fixed_fov is None:
                # GeoCalib will be used
                is_pinhole = camera_type == CameraType.PINHOLE
                if is_pinhole:
                    logger.info("Loading GeoCalib (pinhole)...")
                    cache.geocalib_pinhole = GeoCalib(weights="pinhole").cuda()
                else:
                    logger.info("Loading GeoCalib (distorted)...")
                    cache.geocalib_distorted = GeoCalib(weights="distorted").cuda()
        
        # Load adaptive depth models
        if post_cfg is not None and post_cfg.get("depth_align_model") is not None:
            model_name = post_cfg.depth_align_model
            logger.info(f"Loading adaptive depth models: {model_name}")
            
            try:
                prefix, metric_model, video_model = model_name.split("_")
                assert video_model in ["svda", "vda"]
                
                # Video depth model
                vda_variant = "vits" if video_model == "svda" else "vitl"
                cache.video_depth_model = VideoDepthAnythingDepthModel(model=vda_variant)
                cache.video_depth_model_name = video_model
                
            except ValueError:
                prefix, metric_model = model_name.split("_")
                video_model = None
            
            # Metric depth for adaptive alignment
            cache.adaptive_metric_depth = make_depth_model(metric_model)
            cache.adaptive_metric_depth_name = metric_model
            
            # PriorDA model
            cache.prior_da_model = PriorDAModel()
        
        # Load TrackAnything (SAM + AOT) for instance segmentation
        if init_cfg is not None and init_cfg.get("instance") is not None:
            instance_cfg = init_cfg.instance
            logger.info("Loading TrackAnything (SAM + AOT)...")
            
            # Build mask_phrases including sky if configured
            mask_phrases = list(instance_cfg.get("phrases", []))
            if instance_cfg.get("add_sky", False):
                from vipe.streams.base import VideoFrame
                mask_phrases.append(VideoFrame.SKY_PROMPT)
            
            cache.track_anything = TrackAnythingPipeline(
                mask_phrases=mask_phrases,
                sam_points_per_side=50,  # Match the non-cached version
                sam_run_gap=10,  # Default, will be overridden per video
            )
        
        logger.info("Model cache initialized")
        return cache
    
    def get_metric_depth_for_slam(self, camera_type: CameraType):
        """Get metric depth model adapted for camera type."""
        if self.metric_depth is None:
            return None
        
        if camera_type not in self.metric_depth.supported_camera_types:
            return PinholeDepthAdapter(self.metric_depth)
        return self.metric_depth
    
    def get_geocalib(self, is_pinhole: bool) -> Optional[GeoCalib]:
        """Get appropriate GeoCalib model."""
        if is_pinhole:
            return self.geocalib_pinhole
        return self.geocalib_distorted


# Global cache instance (optional, for convenience)
_global_cache: Optional[VipeModelCache] = None


def get_global_cache() -> Optional[VipeModelCache]:
    """Get the global model cache if initialized."""
    return _global_cache


def set_global_cache(cache: VipeModelCache):
    """Set the global model cache."""
    global _global_cache
    _global_cache = cache


def clear_global_cache():
    """Clear the global model cache."""
    global _global_cache
    _global_cache = None

