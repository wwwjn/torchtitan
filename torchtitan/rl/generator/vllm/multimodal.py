# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Image prompts in the vLLM generator: the TorchTitan vision encoder behind vLLM's
multimodal interface.

Requests carry images as `PromptImage`s (content key, patch grid, placeholder range
and the image file's bytes). ``VLLMEngine`` decodes and resizes each image file with its
`ImageLoader` (``VLLMEngine.Config.images``) and submits vLLM's multimodal input
directly (`image_prompt`), so vLLM's Hugging Face processor never runs. The image's mm hash is its content key, so vLLM's encoder
cache and prefix cache share one image across the samples of a GRPO group:

    PromptImage(key, ..., data)                     rollout worker -> generator: the file's bytes
                                                    (SLAKE median 53 KB)
      -> load_images: uint8 (C, H, W)               each TP rank decodes + resizes the file
                                                    (SLAKE: 1.7 ms median, 6.4 ms 1024x1024)
      -> vLLM mm_input(prompt ids, {"images", "image_grid_thw"}, hashes=key, placeholders)
      -> embed_multimodal: patchify on GPU + vision encoder, once per key while cached
      -> embed_input_ids: token embeddings with image embeddings at the placeholders
      -> forward(inputs_embeds, 3D M-RoPE positions)

Normalizing and patchifying on the GPU instead of the CPU keeps ~20 ms per image off
the engine loop thread and sends 8x fewer bytes host to device; the values are the
ones ``load_pixel_values`` computes on the CPU.

The processor registered with vLLM only serves vLLM's startup: the loader's largest
image (`largest_grid`) sizes the encoder budget, and a dummy image of that size
profiles the encoder's memory.

Supported models: Qwen3.5-style (``Qwen35Model``): a ``vision_encoder`` behind
``get_vision_embeds``, merged patches as placeholder tokens, and ``MRoPE.Config``
attention. `validate_image_model` checks this before the engine is built.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

import torch
from vllm.inputs import mm_input, MultiModalInput
from vllm.model_executor.models.interfaces import SupportsMRoPE, SupportsMultiModal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    InputProcessingContext,
    ProcessorInputs,
    TimingContext,
)

from torchtitan.components.data.multimodal.image_loader import (
    ImageLoader,
    ImageSpec,
    patchify,
)
from torchtitan.components.data.multimodal.mrope import get_mrope_positions
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.qwen3_5.rope import MRoPE
from torchtitan.rl.generator.vllm.model_wrapper import VLLMModelWrapper
from torchtitan.rl.rollout.types import PromptImage

# The keyword arguments vLLM batches per image and passes to `embed_multimodal`.
_IMAGE_FIELDS = {
    # uint8 (C, H, W); images of different sizes stay a list.
    "images": MultiModalFieldConfig.batched("image"),
    # (3,) raw patch grid, stacked to (num_images, 3).
    "image_grid_thw": MultiModalFieldConfig.batched("image"),
}


def validate_image_model(model_config: Decoder.Config, spec: ImageSpec) -> None:
    """Raise unless ``model_config`` can serve images preprocessed with ``spec``.

    Runs before the engine builds the model: image prompts need a vision encoder
    whose patching matches ``spec``, and M-RoPE attention.
    """
    encoder = getattr(model_config, "vision_encoder", None)
    if encoder is None:
        raise ValueError(
            f"{type(model_config).__qualname__} has no vision encoder; build the "
            "generator model with one, or unset VLLMEngine.Config.images."
        )
    if (
        encoder.patch_size,
        encoder.temporal_patch_size,
        encoder.spatial_merge_size,
    ) != (
        spec.patch_size,
        spec.temporal_patch_size,
        spec.merge_size,
    ):
        raise ValueError(
            f"The image loader's spec has patch_size={spec.patch_size}, "
            f"temporal_patch_size={spec.temporal_patch_size}, merge_size={spec.merge_size}; "
            f"the vision encoder uses {encoder.patch_size}, {encoder.temporal_patch_size}, "
            f"{encoder.spatial_merge_size}. Use the model's preprocessing in "
            "ImageLoader.Config.spec."
        )
    mrope_hf_config(model_config)


def mrope_hf_config(model_config: Decoder.Config) -> dict[str, Any]:
    """The HF config entries that make vLLM compute 3D M-RoPE positions.

    With them vLLM calls the model's ``get_mrope_input_positions``; text-only
    generators keep 1D positions.
    """
    attention = model_config.first_full_attention
    rope = getattr(attention, "rope", None)
    if not isinstance(rope, MRoPE.Config):
        raise ValueError(
            "VLLMEngine.Config.images needs a model whose attention uses "
            f"MRoPE.Config (Qwen3.5); got {type(rope).__qualname__}."
        )
    # Presence required: an mrope_section switches vLLM to 3D positions.
    return {
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": rope.theta,
            "mrope_section": list(rope.mrope_section),
        }
    }


def image_prompt(
    prompt_token_ids: list[int],
    images: Sequence[PromptImage],
    pixels: Sequence[torch.Tensor],
    *,
    cache_salt: str | None = None,
) -> MultiModalInput:
    """Build vLLM's multimodal engine input for a prompt whose images are loaded.

    Args:
        prompt_token_ids: Prompt ids with each image's placeholder run expanded.
        images: The prompt's images, in any order.
        pixels: Each image's resized uint8 image ``(C, H, W)``, in ``images`` order.
        cache_salt: vLLM prefix-cache salt.
    """
    kwargs = MultiModalKwargsItems.from_hf_inputs(
        {
            "images": list(pixels),
            "image_grid_thw": torch.tensor([image.grid_thw for image in images]),
        },
        _IMAGE_FIELDS,
    )
    return mm_input(
        prompt_token_ids=prompt_token_ids,
        mm_kwargs=kwargs,
        mm_hashes={"image": [image.key for image in images]},
        mm_placeholders={
            "image": [
                PlaceholderRange(offset=image.offset, length=image.length)
                for image in images
            ]
        },
        cache_salt=cache_salt,
    )


class _ImageProcessingInfo(BaseProcessingInfo):
    """Sizes vLLM's encoder budget and memory profiling: the loader's largest
    image that fits ``max_model_len``."""

    def __init__(self, ctx: InputProcessingContext, *, images: ImageLoader) -> None:
        super().__init__(ctx)
        self.images = images

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def largest_grid(self, seq_len: int) -> tuple[int, int, int]:
        return self.images.largest_grid(seq_len)

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        t, h, w = self.largest_grid(seq_len)
        return {"image": t * h * w // self.images.spec.merge_size**2}


class _ImageDummyInputsBuilder(BaseDummyInputsBuilder[_ImageProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> dict[str, Any]:
        _, h, w = self.info.largest_grid(seq_len)
        patch_size = self.info.images.spec.patch_size
        return {
            "image": self._get_dummy_images(
                width=w * patch_size,
                height=h * patch_size,
                num_images=mm_counts.get("image", 0),
            )
        }


class _ImageProcessor(BaseMultiModalProcessor[_ImageProcessingInfo]):
    """Builds only the dummy inputs vLLM profiles with; real prompts use `image_prompt`."""

    # Prompts arrive as token ids and images preprocessed, so no tokenizer is needed
    # (the engine runs with skip_tokenizer_init).
    requires_tokenizer: ClassVar[bool] = False

    def _get_mm_fields_config(
        self, hf_inputs: Any, hf_processor_mm_kwargs: Mapping[str, object]
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _IMAGE_FIELDS

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs):
        return []

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        """One zero image of the loader's largest size per dummy image item."""
        num_images = inputs.mm_data_items.get_count("image", strict=False)
        grid = self.info.largest_grid(self.info.ctx.model_config.max_model_len)
        spec = self.info.images.spec
        length = grid[0] * grid[1] * grid[2] // spec.merge_size**2
        image = torch.zeros(
            3, grid[1] * spec.patch_size, grid[2] * spec.patch_size, dtype=torch.uint8
        )
        images = [
            PromptImage(
                key=f"profile-{index}",
                path=f"profile-{index}",
                grid_thw=grid,
                offset=index * length,
                length=length,
                data=b"",  # unused: the pixels are `image`
            )
            for index in range(num_images)
        ]
        return image_prompt([0] * (num_images * length), images, [image] * num_images)


class VLLMMultimodalModelWrapper(VLLMModelWrapper, SupportsMultiModal, SupportsMRoPE):
    """`VLLMModelWrapper` that keeps the vision encoder and serves image prompts.

    Implements vLLM's multimodal (`embed_multimodal`, `embed_input_ids`) and M-RoPE
    (`get_mrope_input_positions`) interfaces for Qwen3.5-style models: a
    ``vision_encoder`` reached through the model's ``get_vision_embeds``, merged
    patches as placeholder tokens, and 3D M-RoPE positions. vLLM feeds multimodal
    models embeddings, also for text-only steps, and 3D positions ``(3, num_tokens)``.
    """

    supports_multimodal: ClassVar[Literal[True]] = True

    def __init__(self, *, image_spec: ImageSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._image_spec = image_spec
        # FSDP makes the first module that runs forward its root. This wrapper calls
        # the embeddings, the vision encoder and the layers directly, never the root
        # model, so make the root the whole model before any of them runs. Otherwise
        # the grouped [tok_embeddings, norm, lm_head] state raises ("requires a single
        # root module") when embed_input_ids runs first.
        # TODO: use a public FSDP lazy-init API once one exists.
        self.model._get_fsdp_state()._lazy_init()

    def embed_multimodal(
        self, *, images: torch.Tensor | list[torch.Tensor], image_grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """vLLM required API. Encode a batch of images, one embedding block per image.

        Args:
            images: Resized uint8 images ``(C, H, W)`` on the GPU, stacked when they
                share a size.
            image_grid_thw: ``(num_images, 3)`` raw patch grids.
        """
        pixel_values = patchify(list(images), self._image_spec)
        with self.parallelism_context.activate_spmd():
            vision_embeds, num_tokens = self.model.get_vision_embeds(
                pixel_values, grid_thw=image_grid_thw
            )
        return vision_embeds.split(num_tokens.tolist())

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Sequence[torch.Tensor] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """vLLM required API. Token embeddings with image embeddings written over the
        placeholder tokens that ``is_multimodal`` marks."""
        inputs_embeds = super().embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """vLLM required API. The decoder on precomputed embeddings.

        Args:
            input_ids: Unused; vLLM passes embeddings to multimodal models.
            positions: ``(3, num_tokens)`` M-RoPE positions.
            inputs_embeds: ``(num_tokens, dim)`` from `embed_input_ids`.
        """
        assert inputs_embeds is not None and positions is not None
        # TorchTitan's MRoPE takes (num_tokens, 3).
        positions = positions.transpose(0, 1)
        with self.parallelism_context.activate_spmd():
            # Mirrors the model's forward after embedding (layers, then the final norm).
            # TODO: share this loop with Decoder.forward / Qwen35Model.forward.
            h = inputs_embeds
            layer_kwargs = self._prepare_step_kwargs(h.shape[0])
            for layer in self.model.layers.values():
                h = layer(h, None, positions, **layer_kwargs)
            h = self.model.norm(h)
        return self._replicated_local(h)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        """vLLM required API. ``(3, num_tokens)`` prompt positions and the offset
        ``position - token index`` of every token after the prompt."""
        vision_items = []
        for feature in mm_features:
            # `image_prompt` submits only images, and the processor cache is off, so
            # vLLM keeps each feature's data.
            assert (
                feature.modality == "image" and feature.data is not None
            ), feature.modality
            t, h, w = feature.data["image_grid_thw"].data.tolist()
            vision_items.append((feature.mm_position.offset, (t, h, w)))
        positions = get_mrope_positions(
            len(input_tokens),
            vision_items,
            spatial_merge_size=self._image_spec.merge_size,
        )
        delta = int(positions.max()) + 1 - len(input_tokens)
        return positions.transpose(0, 1), delta


def register_image_processor(model_cls: type, images: ImageLoader) -> None:
    """Register `_ImageProcessor` with vLLM for ``model_cls``, reading ``images``."""
    MULTIMODAL_REGISTRY.register_processor(
        _ImageProcessor,
        info=functools.partial(_ImageProcessingInfo, images=images),
        dummy_inputs=_ImageDummyInputsBuilder,
    )(model_cls)
