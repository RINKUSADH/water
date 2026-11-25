"""
physics_stack.py — turn cached ParamNet outputs into the generator's physics inputs (on GPU).

phys cache channels: [log T1_src, log T2_src, M0, mean log Laplace scale]
Physics stack (5 ch, Phase 3a/3c/concat): [M0, norm logT1_tgt, norm logT2_tgt, Bloch prior at target field, uncertainty]
Prototype input (5 ch, Phase 3b, 2 mm):  [norm logT1_src, norm logT2_src, M0, norm logT1_tgt, norm logT2_tgt]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DEFAULT_SEQ, bloch_signal, norm_log_t1, norm_log_t2  # noqa: E402


def build_physics(phys, src_x, src_t, tgt_t, ft, modality):
    brain = (src_x > 0.005).to(phys.dtype)
    lt1, lt2, m0, logb = phys[:, 0:1], phys[:, 1:2], phys[:, 2:3].clamp(0, 1.5), phys[:, 3:4]
    lt1_t = ft.apply_log(lt1, "t1", src_t, tgt_t)
    lt2_t = ft.apply_log(lt2, "t2", src_t, tgt_t)
    prior = bloch_signal(m0, lt1_t.exp(), lt2_t.exp(), modality, DEFAULT_SEQ[modality]) * brain
    ref = torch.quantile(prior.flatten(1)[:, ::17].float(), 0.99, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1, 1)
    prior = (prior / ref.to(prior.dtype)).clamp(0, 1)
    unc = torch.sigmoid(logb + 2.0)
    stack = torch.cat([m0 / 1.5, norm_log_t1(lt1_t), norm_log_t2(lt2_t), prior, unc], 1)
    return stack * brain


def build_proto_input(phys_ctx, ctx, src_t, tgt_t, ft):
    lt1, lt2, m0 = phys_ctx[:, 0:1], phys_ctx[:, 1:2], phys_ctx[:, 2:3].clamp(0, 1.5)
    lt1_t = ft.apply_log(lt1, "t1", src_t, tgt_t)
    lt2_t = ft.apply_log(lt2, "t2", src_t, tgt_t)
    pc = torch.cat([norm_log_t1(lt1), norm_log_t2(lt2), m0 / 1.5, norm_log_t1(lt1_t), norm_log_t2(lt2_t)], 1)
    return pc, ctx > 0.01
