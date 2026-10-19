# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampler that draws the same tokens and logprobs as the vLLM generator.

Shape legend: B = requests in the step, V = vocabulary size.

The RL generator samples with full-vocabulary temperature sampling (``top_p`` must
be 1.0, see ``SamplingConfig``) and returns the logprob of the sampled token after
temperature scaling (vLLM's ``processed_logprobs``). This sampler reproduces vLLM's
v1 sampler for that case step for step:

- logits are cast to fp32;
- greedy rows take ``argmax`` of the unscaled logits;
- other rows divide the logits by their temperature, take ``log_softmax`` for the
  logprobs and ``softmax`` for the probabilities, and pick ``argmax(probs / q)``
  with exponential noise ``q``;
- a request with a seed draws its noise row from its own CUDA generator seeded
  once at its first sampled token, so its tokens do not depend on the other
  requests in the step. A partially prefilled request draws nothing, as vLLM
  rewinds the generator of a token it discards.

Given bitwise-identical logits, both engines therefore return bitwise-identical
tokens and logprobs.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.layers.sampler import Sampler


class TorchTitanSampler(Sampler):
    """SGLang ``Sampler`` with the vLLM v1 sampling semantics described above.

    The engine publishes the step's requests in row order through ``set_batch_reqs``
    before the forward, and releases the per-request generators of finished requests
    through ``release``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._generators: dict[str, torch.Generator] = {}
        self._batch_reqs: list | None = None

    def set_batch_reqs(self, reqs: list) -> None:
        self._batch_reqs = reqs

    def release(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            self._generators.pop(request_id, None)

    def _generator(self, req, device: torch.device) -> torch.Generator:
        generator = self._generators.get(req.rid)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(req.sampling_params.sampling_seed)
            self._generators[req.rid] = generator
        return generator

    def forward(
        self,
        logits_output,
        sampling_info,
        return_logprob: bool,
        top_logprobs_nums: list[int],
        token_ids_logprobs: list[list[int]],
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            raise NotImplementedError(
                f"TorchTitanSampler does not support {sorted(kwargs)}"
            )
        reqs = self._batch_reqs
        logits_BV = logits_output.next_token_logits.to(torch.float32)
        # Data-parallel attention may pad a step with rows after the real requests'
        # (SGLang drops their tokens); they sample greedily.
        assert (
            reqs is not None and len(reqs) <= logits_BV.shape[0]
        ), "The engine must publish the step's requests before sampling"
        if sampling_info.has_custom_logit_processor:
            raise NotImplementedError("custom logit processors are not supported")
        num_padding_rows = logits_BV.shape[0] - len(reqs)

        # SGLang represents a greedy request (temperature 0) as top_k == 1.
        greedy_B = torch.nn.functional.pad(
            sampling_info.top_ks == 1, (0, num_padding_rows), value=True
        )
        if sampling_info.is_all_greedy:
            next_token_ids_B = logits_BV.argmax(dim=-1)
            logprobs_BV = logits_BV.log_softmax(dim=-1, dtype=torch.float32)
        else:
            greedy_ids_B = logits_BV.argmax(dim=-1)
            # SGLang already stores temperature 1.0 for greedy rows.
            logits_BV.div_(
                torch.nn.functional.pad(
                    sampling_info.temperatures, (0, 0, 0, num_padding_rows), value=1.0
                )
            )
            logprobs_BV = logits_BV.log_softmax(dim=-1, dtype=torch.float32)
            probs_BV = logits_BV.softmax(dim=-1, dtype=torch.float32)
            noise_BV = torch.empty_like(probs_BV)
            seeded_rows = [
                i
                for i, (req, greedy) in enumerate(zip(reqs, greedy_B.tolist()))
                if not greedy
                and req.sampling_params.sampling_seed is not None
                and req.inflight_middle_chunks <= 0
            ]
            if len(seeded_rows) != logits_BV.shape[0]:
                noise_BV.exponential_()
            for i in seeded_rows:
                noise_BV[i].exponential_(
                    generator=self._generator(reqs[i], logits_BV.device)
                )
            random_ids_B = probs_BV.div_(noise_BV).argmax(dim=-1)
            next_token_ids_B = torch.where(greedy_B, greedy_ids_B, random_ids_B)

        if return_logprob:
            self.output_logprob_processor.compute_logprobs(
                logprobs_BV,
                top_logprobs_nums,
                token_ids_logprobs,
                next_token_ids_B,
            ).write_output_to(logits_output)
        return next_token_ids_B
