# Copyright 2024 PRIME team and/or its affiliates
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
"""Core algorithms for the RFT recipe."""

from __future__ import annotations

import torch

import verl
import verl.utils.torch_functional as verl_F


def compute_gated_token_reward(
    teacher_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    verifier_scores: torch.Tensor,
    entropys: torch.Tensor,
    config,
) -> torch.Tensor:
    """Compute gated token-level reward.

    Parameters
    ----------
    teacher_log_probs : torch.Tensor
        Log probabilities from the teacher model of shape ``(bsz, seq_len)``.
    ref_log_probs : torch.Tensor
        Log probabilities from the frozen reference model.
    verifier_scores : torch.Tensor
        Sequence level verifier score in the range ``[0, 1]``.
    entropys : torch.Tensor
        Token entropy from the actor model.
    config : Any
        Configuration object that contains ``token_gate`` fields ``k_r``,
        ``k_H``, ``tau_r`` and ``tau_H``.
    """

    k_r = config.algorithm.token_gate.k_r
    tau_r = config.algorithm.token_gate.tau_r
    k_h = config.algorithm.token_gate.k_H
    tau_h = config.algorithm.token_gate.tau_H

    gate_r = torch.sigmoid(k_r * (tau_r - verifier_scores)).unsqueeze(-1)
    gate_h = torch.sigmoid(k_h * (tau_h - entropys)).unsqueeze(-1)

    return (teacher_log_probs - ref_log_probs) * gate_r * gate_h


def compute_rft_advantage_return(data: verl.DataProto, response_mask: torch.Tensor, n_samples: int, config):
    """Compute advantages using the gated token reward."""

    teacher = data.batch["teacher_log_probs"]
    ref = data.batch["ref_log_probs"]
    verifier = data.batch["acc"]
    entropys = data.batch["entropys"]

    token_reward = compute_gated_token_reward(teacher, ref, verifier, entropys, config)

    with torch.no_grad():
        returns = (token_reward * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = verl_F.masked_whiten(returns, response_mask)

    return advantages, returns
