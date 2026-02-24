from typing import Dict

import torch
from torch import nn

from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "dynamics_predictor")
def build(config, name: str):
    return None


class DynamicsPredictor(nn.Module):
    def __init__(self, history_len: int):
        super().__init__()
        self.history_len = history_len

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("This method should be implemented in the child class.")


class SlotFormer(DynamicsPredictor):
    """Implementation of the SlotFormer model.
    Adapted from the official implementation: https://github.com/pairlab/SlotFormer/blob/master/slotformer/video_prediction/models/slotformer.py
    Link to the paper: https://arxiv.org/abs/2210.05861"""

    def __init__(
        self,
        num_slots=7,
        slot_size=128,
        history_len=6,
        t_pe="sin",
        slots_pe="",
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
        norm_first=True,
    ):
        super().__init__(history_len)

        self.num_slots = num_slots
        # Projection to the latent space
        self.in_proj = nn.Linear(slot_size, d_model)
        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers
        )
        # Positional encodings
        self.enc_t_pe = self._build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = self._build_pos_enc(slots_pe, num_slots, d_model)
        # Projection to the output space
        self.out_proj = nn.Linear(d_model, slot_size)

    def _build_pos_enc(self, pos_enc, input_len, d_model):
        """Build positional encodings."""
        if not pos_enc:
            return None
        if pos_enc == "learnable":
            pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
        elif "sin" in pos_enc:
            inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
            pos_seq = torch.arange(input_len - 1, -1, -1).type_as(inv_freq)
            sinusoid_inp = torch.outer(pos_seq, inv_freq)
            pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
            pos_embedding = nn.Parameter(pos_emb.unsqueeze(0), requires_grad=False)
        else:
            raise NotImplementedError(f"{pos_enc} is not supported as a positional encoding.")
        return pos_embedding

    def forward(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        past_slots = slots[:, : self.history_len]  # [B, T, N, slot_size]
        rollout_len = slots.shape[1] - self.history_len
        # Add positional encodings
        past_slots = past_slots.flatten(1, 2)
        in_slots = past_slots
        enc_pe = (
            self.enc_t_pe.unsqueeze(2)
            .repeat(past_slots.shape[0], 1, self.num_slots, 1)
            .flatten(1, 2)
        )
        if self.enc_slots_pe is not None:
            slots_pe = (
                self.enc_slots_pe.unsqueeze(1)
                .repeat(past_slots.shape[0], self.history_len, 1, 1)
                .flatten(1, 2)
            )
            enc_pe = slots_pe + enc_pe
        # Autoregressive prediction
        pred_out = []
        for _ in range(rollout_len):
            past_slots = self.in_proj(in_slots)
            past_slots = past_slots + enc_pe
            past_slots = self.transformer_encoder(past_slots)
            pred_slots = self.out_proj(past_slots[:, -self.num_slots :])
            pred_out.append(pred_slots)
            in_slots = torch.cat([in_slots[:, self.num_slots :], pred_out[-1]], dim=1)
        # Stack the predictions
        pred_slots = torch.stack(pred_out, dim=1)
        return {"next_state": pred_slots}