import torch
from torch import nn


class AgentCausalSlotPredictor(nn.Module):
    """Agent-centric slot predictor with explicit history/proprio conditioning.

    Assumptions:
    - The last slot is the agent slot.
    - Actions directly affect the agent transition.
    - Object transitions are conditioned on the current object state together with
      the current/next agent state and the predicted agent delta.
    """

    def __init__(
        self,
        slot_dim: int,
        action_dim: int,
        proprio_dim: int = 0,
        history_len: int = 3,
        history_layers: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        stop_gradient_agent_to_object: bool = True,
        use_agent_delta: bool = True,
        use_object_self_attn: bool = True,
    ):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.history_len = max(1, int(history_len))
        self.stop_gradient_agent_to_object = bool(stop_gradient_agent_to_object)
        self.use_agent_delta = bool(use_agent_delta)
        self.use_object_self_attn = bool(use_object_self_attn)

        self.action_proj = nn.Linear(self.action_dim, self.slot_dim)
        self.proprio_in_proj = nn.Linear(self.proprio_dim, self.slot_dim) if self.proprio_dim > 0 else None
        self.proprio_pred_proj = nn.Linear(self.slot_dim, self.proprio_dim) if self.proprio_dim > 0 else None
        self.delta_agent_proj = nn.Linear(self.slot_dim, self.slot_dim)
        self.delta_proprio_proj = nn.Linear(self.slot_dim, self.slot_dim) if self.proprio_dim > 0 else None

        hist_in_dim = self.slot_dim * 2 if self.proprio_dim > 0 else self.slot_dim
        self.history_token_proj = nn.Linear(hist_in_dim, self.slot_dim)
        self.history_pos_embed = nn.Parameter(torch.zeros(1, self.history_len, self.slot_dim))
        hist_layer = nn.TransformerEncoderLayer(
            d_model=self.slot_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(hist_layer, num_layers=max(1, int(history_layers)))
        self.history_norm = nn.LayerNorm(self.slot_dim)

        agent_query_dim = self.slot_dim * 4
        self.agent_query_proj = nn.Linear(agent_query_dim, self.slot_dim)
        self.agent_q_norm = nn.LayerNorm(self.slot_dim)
        self.agent_kv_norm = nn.LayerNorm(self.slot_dim)
        self.agent_cross_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.agent_ff_norm = nn.LayerNorm(self.slot_dim)
        self.agent_ff = nn.Sequential(
            nn.Linear(self.slot_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, self.slot_dim),
            nn.Dropout(dropout),
        )

        self.obj_q_norm = nn.LayerNorm(self.slot_dim)
        self.obj_kv_norm = nn.LayerNorm(self.slot_dim)
        self.obj_cross_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.obj_ff_norm = nn.LayerNorm(self.slot_dim)
        self.obj_ff = nn.Sequential(
            nn.Linear(self.slot_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, self.slot_dim),
            nn.Dropout(dropout),
        )

        if self.use_object_self_attn:
            self.obj_self_norm = nn.LayerNorm(self.slot_dim)
            self.obj_self_attn = nn.MultiheadAttention(
                embed_dim=self.slot_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.obj_self_ff_norm = nn.LayerNorm(self.slot_dim)
            self.obj_self_ff = nn.Sequential(
                nn.Linear(self.slot_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, self.slot_dim),
                nn.Dropout(dropout),
            )

    def _crop_history(self, x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.size(1) <= self.history_len:
            return x
        return x[:, -self.history_len :, ...]

    def _encode_agent_history(
        self,
        agent_history: torch.Tensor,
        proprio_history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if proprio_history is not None:
            proprio_ctx = self.proprio_in_proj(proprio_history)
            hist_tokens = torch.cat([agent_history, proprio_ctx], dim=-1)
            current_prop = proprio_ctx[:, -1:, :]
        else:
            hist_tokens = agent_history
            current_prop = agent_history.new_zeros(agent_history.size(0), 1, self.slot_dim)

        hist_tokens = self.history_token_proj(hist_tokens)
        hist_tokens = self._crop_history(hist_tokens)
        pos = self.history_pos_embed[:, : hist_tokens.size(1), :]
        hist_tokens = self.history_encoder(hist_tokens + pos)
        summary = self.history_norm(hist_tokens[:, -1:, :])
        return summary, current_prop

    def _predict_one_step(
        self,
        slot_history: torch.Tensor,
        action_t: torch.Tensor,
        proprio_history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Args:
            slot_history: (B, T_hist, S, D)
            action_t: (B, A)
            proprio_history: (B, T_hist, P_emb)
        Returns:
            slots_next: (B, S, D)
            non_agent_next: (B, S-1, D)
            agent_next: (B, D)
            pred_proprio_embed: (B, P_emb) or None
            agent_delta: (B, D)
        """
        if slot_history.ndim != 4:
            raise ValueError(f"slot_history must be (B, T_hist, S, D), got {tuple(slot_history.shape)}")
        if action_t.ndim != 2:
            raise ValueError(f"action_t must be (B, A), got {tuple(action_t.shape)}")
        if slot_history.size(2) < 2:
            raise ValueError("Need at least 2 slots (one object + one agent).")

        slots_t = slot_history[:, -1, :, :]
        non_agent_t = slots_t[:, :-1, :]
        agent_t = slots_t[:, -1:, :]
        agent_history = slot_history[:, :, -1, :]

        history_summary, current_prop = self._encode_agent_history(agent_history, proprio_history)
        action_embed = self.action_proj(action_t).unsqueeze(1)
        agent_query = self.agent_query_proj(torch.cat([agent_t, history_summary, action_embed, current_prop], dim=-1))

        agent_context = [slots_t, history_summary, current_prop]
        kv_agent = self.agent_kv_norm(torch.cat(agent_context, dim=1))
        q_agent = self.agent_q_norm(agent_query)
        delta_agent_attn, _ = self.agent_cross_attn(q_agent, kv_agent, kv_agent, need_weights=False)
        agent_next = agent_t + delta_agent_attn
        agent_next = agent_next + self.agent_ff(self.agent_ff_norm(agent_next))
        agent_delta = (agent_next - agent_t).squeeze(1)

        pred_proprio_embed = None
        pred_proprio_delta = None
        if self.proprio_pred_proj is not None:
            pred_proprio_embed = self.proprio_pred_proj(agent_next.squeeze(1))
            pred_proprio_ctx = pred_proprio_embed.unsqueeze(1)
            pred_proprio_delta = pred_proprio_ctx - current_prop
        else:
            pred_proprio_ctx = history_summary.new_zeros(history_summary.size(0), 1, self.slot_dim)

        if self.stop_gradient_agent_to_object:
            agent_t_for_objects = agent_t.detach()
            agent_next_for_objects = agent_next.detach()
            agent_delta_for_objects = agent_delta.detach()
        else:
            agent_t_for_objects = agent_t
            agent_next_for_objects = agent_next
            agent_delta_for_objects = agent_delta

        obj_context = [non_agent_t, agent_t_for_objects, agent_next_for_objects, history_summary]
        if self.use_agent_delta:
            obj_context.append(self.delta_agent_proj(agent_delta_for_objects).unsqueeze(1))
        if self.proprio_in_proj is not None:
            obj_context.append(current_prop)
            obj_context.append(pred_proprio_ctx)
            obj_context.append(self.delta_proprio_proj(pred_proprio_delta))

        kv_obj = self.obj_kv_norm(torch.cat(obj_context, dim=1))
        q_obj = self.obj_q_norm(non_agent_t)
        delta_obj, _ = self.obj_cross_attn(q_obj, kv_obj, kv_obj, need_weights=False)
        non_agent_next = non_agent_t + delta_obj
        non_agent_next = non_agent_next + self.obj_ff(self.obj_ff_norm(non_agent_next))

        if self.use_object_self_attn:
            obj_self, _ = self.obj_self_attn(
                self.obj_self_norm(non_agent_next),
                self.obj_self_norm(non_agent_next),
                self.obj_self_norm(non_agent_next),
                need_weights=False,
            )
            non_agent_next = non_agent_next + obj_self
            non_agent_next = non_agent_next + self.obj_self_ff(self.obj_self_ff_norm(non_agent_next))

        slots_next = torch.cat([non_agent_next, agent_next], dim=1)
        return slots_next, non_agent_next, agent_next.squeeze(1), pred_proprio_embed, agent_delta

    def rollout(
        self,
        history_slots: torch.Tensor,
        action_seq: torch.Tensor,
        history_proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            history_slots: (B, T_hist, S, D)
            action_seq: (B, T, A)
            history_proprio: (B, T_hist, P_emb)
        Returns:
            pred_slots: (B, T, S, D)
            aux:
              - pred_object_slots: (B, T, S-1, D)
              - pred_agent_slots: (B, T, D)
              - pred_proprio_embeds: (B, T, P_emb) or None
              - pred_agent_deltas: (B, T, D)
        """
        if history_slots.ndim != 4:
            raise ValueError(f"history_slots must be (B, T_hist, S, D), got {tuple(history_slots.shape)}")
        if action_seq.ndim != 3:
            raise ValueError(f"action_seq must be (B, T, A), got {tuple(action_seq.shape)}")
        if history_slots.size(0) != action_seq.size(0):
            raise ValueError("Batch size mismatch between history_slots and action_seq.")
        if history_proprio is not None and history_proprio.size(0) != history_slots.size(0):
            raise ValueError("Batch size mismatch between history_slots and history_proprio.")

        slot_hist = self._crop_history(history_slots)
        proprio_hist = self._crop_history(history_proprio)

        pred_all = []
        pred_obj = []
        pred_agent = []
        pred_prop = []
        pred_delta = []
        for t in range(action_seq.size(1)):
            slots_next, obj_next, agent_next, proprio_next, agent_delta = self._predict_one_step(
                slot_hist,
                action_seq[:, t, :],
                proprio_hist,
            )
            pred_all.append(slots_next)
            pred_obj.append(obj_next)
            pred_agent.append(agent_next)
            pred_delta.append(agent_delta)
            if proprio_next is not None:
                pred_prop.append(proprio_next)

            slot_hist = torch.cat([slot_hist, slots_next.unsqueeze(1)], dim=1)
            slot_hist = self._crop_history(slot_hist)
            if proprio_hist is not None and proprio_next is not None:
                proprio_hist = torch.cat([proprio_hist, proprio_next.unsqueeze(1)], dim=1)
                proprio_hist = self._crop_history(proprio_hist)

        if len(pred_all) == 0:
            bsz, _, num_slots, dim = history_slots.shape
            pred_future_slots = history_slots.new_empty((bsz, 0, num_slots, dim))
            pred_object_slots = history_slots.new_empty((bsz, 0, num_slots - 1, dim))
            pred_agent_slots = history_slots.new_empty((bsz, 0, dim))
            pred_agent_deltas = history_slots.new_empty((bsz, 0, dim))
            pred_proprio_embeds = None
            if self.proprio_dim > 0:
                pred_proprio_embeds = history_slots.new_empty((bsz, 0, self.proprio_dim))
        else:
            pred_future_slots = torch.stack(pred_all, dim=1)
            pred_object_slots = torch.stack(pred_obj, dim=1)
            pred_agent_slots = torch.stack(pred_agent, dim=1)
            pred_agent_deltas = torch.stack(pred_delta, dim=1)
            pred_proprio_embeds = torch.stack(pred_prop, dim=1) if pred_prop else None

        aux = {
            "pred_object_slots": pred_object_slots,
            "pred_agent_slots": pred_agent_slots,
            "pred_agent_deltas": pred_agent_deltas,
            "pred_proprio_embeds": pred_proprio_embeds,
        }
        return pred_future_slots, aux

    def forward(
        self,
        history_slots: torch.Tensor,
        future_actions: torch.Tensor,
        history_proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        if history_slots.ndim != 4:
            raise ValueError(f"history_slots must be (B, T_hist, S, D), got {tuple(history_slots.shape)}")
        if future_actions.ndim != 3:
            raise ValueError(f"future_actions must be (B, T_pred, A), got {tuple(future_actions.shape)}")
        if history_slots.size(0) != future_actions.size(0):
            raise ValueError("Batch size mismatch between history_slots and future_actions.")
        return self.rollout(history_slots, future_actions, history_proprio=history_proprio)

    @torch.no_grad()
    def inference(
        self,
        history_slots: torch.Tensor,
        future_actions: torch.Tensor,
        history_proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_future_slots, _ = self(history_slots, future_actions, history_proprio=history_proprio)
        return pred_future_slots
