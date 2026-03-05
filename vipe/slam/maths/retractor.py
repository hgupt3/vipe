# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from vipe.ext.lietorch import SE3
from vipe.utils.cameras import CameraType


class BaseRetractor:
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor):
        x[inds] += dx


class PoseRetractor(BaseRetractor):
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor):
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class RigRotationOnlyRetractor(BaseRetractor):
    def oplus(self, x: SE3, inds: torch.Tensor, dx: torch.Tensor):
        dx = dx.clone()
        dx[:, :3] = 0  # zero out translation part
        x.data[inds] = SE3(x.data[inds]).retr(dx).data


class DenseDispRetractor(BaseRetractor):
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor):
        dx = torch.where(dx > 10, torch.zeros_like(dx), dx)
        return super().oplus(x, inds, dx)


class TracksDispRetractor(BaseRetractor):
    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor):
        super().oplus(x, inds, dx)
        x.clamp_(min=1e-3, max=10)


class IntrinsicsRetractor(BaseRetractor):
    def __init__(self, camera_type: CameraType, image_height: int = None, fov_min_degrees: float = None, fov_max_degrees: float = None):
        self.camera_type = camera_type
        self.image_height = image_height
        # Convert FOV bounds to focal length bounds (if specified)
        # fy = height / (2 * tan(fov_y / 2))
        # Higher FOV = lower focal length, so min FOV -> max focal, max FOV -> min focal
        if image_height is not None and fov_max_degrees is not None:
            self.fy_min = image_height / (2 * torch.tan(torch.tensor(fov_max_degrees * 3.14159 / 180 / 2)))
        else:
            self.fy_min = None
        if image_height is not None and fov_min_degrees is not None:
            self.fy_max = image_height / (2 * torch.tan(torch.tensor(fov_min_degrees * 3.14159 / 180 / 2)))
        else:
            self.fy_max = None

    def oplus(self, x: torch.Tensor, inds: torch.Tensor, dx: torch.Tensor):
        if len(dx) == 1:
            # Broadcast dx to all intrinsics
            inds = torch.where(x[:, 0] > 0)[0]
            dx = dx.repeat(len(inds), 1)
        x[inds, :2] += dx[..., :1]
        # Use smaller learning rate for the distortion parameters
        x[inds, 4:] += dx[..., 1:] * 0.01
        
        # Clamp focal length to FOV bounds if specified
        if self.fy_min is not None or self.fy_max is not None:
            fy_min = self.fy_min.to(x.device) if self.fy_min is not None else None
            fy_max = self.fy_max.to(x.device) if self.fy_max is not None else None
            if fy_min is not None and fy_max is not None:
                x[inds, :2] = x[inds, :2].clamp(min=fy_min.item(), max=fy_max.item())
            elif fy_min is not None:
                x[inds, :2] = x[inds, :2].clamp(min=fy_min.item())
            elif fy_max is not None:
                x[inds, :2] = x[inds, :2].clamp(max=fy_max.item())
