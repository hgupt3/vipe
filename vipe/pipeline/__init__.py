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


import copy
import importlib
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from omegaconf import DictConfig

from vipe.streams.base import MultiviewVideoList, VideoStream

if TYPE_CHECKING:
    from vipe.model_cache import VipeModelCache


@dataclass(kw_only=True, slots=True)
class AnnotationPipelineOutput:
    # Eager return of the payload values that comes from early stages of the pipeline.
    payload: Any | None = None
    output_streams: Sequence[VideoStream] | None = None


class Pipeline(ABC):
    def __init__(self) -> None:
        self._return_payload = False
        self._return_output_streams = False

    @property
    def return_payload(self) -> bool:
        return self._return_payload

    @return_payload.setter
    def return_payload(self, value: bool) -> None:
        assert isinstance(value, bool), "return_payload must be a boolean"
        self._return_payload = value
        if value:
            self._return_output_streams = False

    @property
    def return_output_streams(self) -> bool:
        return self._return_output_streams

    @return_output_streams.setter
    def return_output_streams(self, value: bool) -> None:
        assert isinstance(value, bool), "return_output_streams must be a boolean"
        self._return_output_streams = value
        if value:
            self._return_payload = False

    def should_filter(self, stream_name: str) -> bool:
        return False

    @abstractmethod
    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput: ...


def make_pipeline_cls(config: DictConfig) -> type[Pipeline]:
    module_path, class_name = config.instance.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def make_pipeline(
    config: DictConfig, 
    model_cache: "VipeModelCache | None" = None,
) -> Pipeline:
    """
    Create a pipeline from config.
    
    Args:
        config: Pipeline configuration
        model_cache: Optional model cache for reusing expensive models across videos
    """
    config = copy.deepcopy(config)
    pipeline_cls = make_pipeline_cls(config)
    del config.instance
    return pipeline_cls(**config, model_cache=model_cache)
