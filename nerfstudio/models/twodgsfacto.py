# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
2D Gaussian Splatting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization_2dgs  # type: ignore
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class TwoDGSfactoModelConfig(SplatfactoModelConfig):
    """2DGS Splatfacto Model Config"""

    _target: Type = field(default_factory=lambda: TwoDGSfactoModel)
    normal_loss_lambda: float = 0.1
    """Lambda for normal consistency loss."""
    use_absgrad: bool = False
    """Whether to use absgrad for densification. Not supported for 2DGS."""


class TwoDGSfactoModel(SplatfactoModel):
    """2DGS Splatfacto Model"""

    config: TwoDGSfactoModelConfig

    def populate_modules(self):
        """Set up the model."""
        super().populate_modules()
        from gsplat.strategy import DefaultStrategy

        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.key_for_gradient = "gradient_2dgs"

    def get_outputs(self, camera: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs."""
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)
            sh_degree_to_use = None

        (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            self.info,
        ) = rasterization_2dgs(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree_to_use,
            packed=False,
            absgrad=self.config.use_absgrad,
            sparse_grad=False,
        )

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )

        background = self._get_background_color()
        rgb = render_colors + (1 - render_alphas) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        depth_im = render_median.squeeze(0) if render_median is not None else None
        if depth_im is not None:
            depth_im = torch.where(render_alphas.squeeze(0) > 0, depth_im, depth_im.detach().max())
        return {
            "rgb": rgb.squeeze(0),
            "accumulation": render_alphas.squeeze(0),
            "depth": depth_im,
            "background": background,
            "normals": render_normals.squeeze(0) if render_normals is not None else torch.zeros_like(rgb),
            "normals_from_depth": normals_from_depth.squeeze(0) if normals_from_depth is not None else torch.zeros_like(rgb),
        }

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)

        if self.training and self.config.normal_loss_lambda > 0.0:
            normals = outputs["normals"]
            normals_from_depth = outputs["normals_from_depth"]
            alphas = outputs["accumulation"]

            if normals is not None and normals.shape[0] != 0:
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normal_loss = self.config.normal_loss_lambda * normal_error.mean()
                loss_dict["normal_loss"] = normal_loss

        return loss_dict
