import torch
from torch import nn


class MultiViewAgentCausalSlotPredictor(nn.Module):
    """Multi-view agent-centric transition model.

    Input format:
    - slots are concatenated on slot axis: (B, V*S, D)
    - each view has fixed slot order and the last slot is the agent slot
    """

    def __init__(
        self,
        slot_dim: int,
        action_dim: int,
        num_views: int,
        slots_per_view: int,
        num_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        stop_gradient_agent_to_object: bool = True,
        use_view_pe: bool = True,
        use_slot_pe: bool = True,
    ):
        super().__init__()
        if num_views < 2:
            raise ValueError(f"num_views must be >= 2, got {num_views}")
        if slots_per_view < 2:
            raise ValueError(f"slots_per_view must be >= 2, got {slots_per_view}")

        self.slot_dim = int(slot_dim)
        self.action_dim = int(action_dim)
        self.num_views = int(num_views)
        self.slots_per_view = int(slots_per_view)
        self.stop_gradient_agent_to_object = bool(stop_gradient_agent_to_object)
        self.use_view_pe = bool(use_view_pe)
        self.use_slot_pe = bool(use_slot_pe)

        self.action_proj = nn.Linear(self.action_dim, self.slot_dim)
        self.agent_query_proj = nn.Linear(self.slot_dim * 2, self.slot_dim)
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

        self.view_embedding = nn.Embedding(self.num_views, self.slot_dim) if self.use_view_pe else None
        self.slot_embedding = (
            nn.Embedding(self.slots_per_view, self.slot_dim) if self.use_slot_pe else None
        )

    def _split_views(self, slots: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3:
            raise ValueError(f"slots must be (B, V*S, D), got {tuple(slots.shape)}")
        expected_slots = self.num_views * self.slots_per_view
        if slots.size(1) != expected_slots:
            raise ValueError(
                f"slots shape mismatch: got {slots.size(1)} slots, expected {expected_slots} "
                f"(num_views={self.num_views}, slots_per_view={self.slots_per_view})"
            )
        return slots.reshape(slots.size(0), self.num_views, self.slots_per_view, self.slot_dim)

    @staticmethod
    def _flatten_views(slots_view: torch.Tensor) -> torch.Tensor:
        # (B, V, S, D) -> (B, V*S, D)
        return slots_view.reshape(slots_view.size(0), -1, slots_view.size(-1))

    def _get_view_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.view_embedding is None:
            return torch.zeros((1, self.num_views, 1, self.slot_dim), device=device, dtype=dtype)
        view_ids = torch.arange(self.num_views, device=device)
        return self.view_embedding(view_ids).view(1, self.num_views, 1, self.slot_dim).to(dtype)

    def _get_slot_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.slot_embedding is None:
            return torch.zeros((1, 1, self.slots_per_view, self.slot_dim), device=device, dtype=dtype)
        slot_ids = torch.arange(self.slots_per_view, device=device)
        return self.slot_embedding(slot_ids).view(1, 1, self.slots_per_view, self.slot_dim).to(dtype)

    def _predict_one_step(
        self,
        slots_t: torch.Tensor,
        action_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slots_t: (B, V*S, D)
            action_t: (B, A)
        Returns:
            slots_next: (B, V*S, D)
            objects_next: (B, V, S-1, D)
            agents_next: (B, V, D)
        """
        if action_t.ndim != 2:
            raise ValueError(f"action_t must be (B, A), got {tuple(action_t.shape)}")
        if slots_t.size(0) != action_t.size(0):
            raise ValueError("Batch size mismatch between slots_t and action_t.")

        slots_view = self._split_views(slots_t)  # (B, V, S, D)
        bsz = slots_view.size(0)
        view_bias = self._get_view_bias(slots_view.device, slots_view.dtype)
        slot_bias = self._get_slot_bias(slots_view.device, slots_view.dtype)
        slots_view_ctx = slots_view + view_bias + slot_bias

        all_slots_ctx = self._flatten_views(slots_view_ctx)  # (B, V*S, D)
        kv_agent = self.agent_kv_norm(all_slots_ctx)
        action_embed = self.action_proj(action_t).unsqueeze(1)  # (B, 1, D)

        agent_raw_list = []
        agent_next_list = []
        for v in range(self.num_views):
            agent_t_raw = slots_view[:, v, -1:, :]  # (B, 1, D)
            agent_t_ctx = slots_view_ctx[:, v, -1:, :]
            agent_query = self.agent_query_proj(torch.cat([agent_t_ctx, action_embed], dim=-1))
            q_agent = self.agent_q_norm(agent_query)
            delta_agent, _ = self.agent_cross_attn(q_agent, kv_agent, kv_agent, need_weights=False)
            agent_next = agent_t_raw + delta_agent
            agent_next = agent_next + self.agent_ff(self.agent_ff_norm(agent_next))
            agent_raw_list.append(agent_t_raw)
            agent_next_list.append(agent_next)

        agents_t = torch.stack(agent_raw_list, dim=1)  # (B, V, 1, D)
        agents_next = torch.stack(agent_next_list, dim=1)  # (B, V, 1, D)
        if self.stop_gradient_agent_to_object:
            agents_t_obj = agents_t.detach()
            agents_next_obj = agents_next.detach()
        else:
            agents_t_obj = agents_t
            agents_next_obj = agents_next

        non_agent_t = slots_view[:, :, :-1, :]  # (B, V, S-1, D)
        non_agent_ctx = slots_view_ctx[:, :, :-1, :]
        non_agent_ctx_flat = non_agent_ctx.reshape(bsz, -1, self.slot_dim)  # (B, V*(S-1), D)

        agent_slot_bias = slot_bias[:, :, -1:, :]
        agents_t_obj_ctx = agents_t_obj + view_bias + agent_slot_bias
        agents_next_obj_ctx = agents_next_obj + view_bias + agent_slot_bias
        kv_obj = torch.cat(
            [
                non_agent_ctx_flat,
                agents_t_obj_ctx.reshape(bsz, -1, self.slot_dim),
                agents_next_obj_ctx.reshape(bsz, -1, self.slot_dim),
            ],
            dim=1,
        )
        kv_obj = self.obj_kv_norm(kv_obj)

        objects_next_list = []
        for v in range(self.num_views):
            non_agent_v_raw = non_agent_t[:, v, :, :]
            non_agent_v_ctx = non_agent_ctx[:, v, :, :]
            q_obj = self.obj_q_norm(non_agent_v_ctx)
            delta_obj, _ = self.obj_cross_attn(q_obj, kv_obj, kv_obj, need_weights=False)
            obj_next_v = non_agent_v_raw + delta_obj
            obj_next_v = obj_next_v + self.obj_ff(self.obj_ff_norm(obj_next_v))
            objects_next_list.append(obj_next_v)

        objects_next = torch.stack(objects_next_list, dim=1)  # (B, V, S-1, D)
        agents_next_flat = agents_next.squeeze(2)  # (B, V, D)

        view_slots_next = []
        for v in range(self.num_views):
            slots_next_v = torch.cat([objects_next[:, v, :, :], agents_next[:, v, :, :]], dim=1)
            view_slots_next.append(slots_next_v)
        slots_next = torch.stack(view_slots_next, dim=1).reshape(bsz, -1, self.slot_dim)

        return slots_next, objects_next, agents_next_flat

    def rollout(
        self,
        start_slots: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            start_slots: (B, V*S, D)
            action_seq: (B, T, A)
        Returns:
            pred_slots: (B, T, V*S, D)
            aux:
              - pred_object_slots: (B, T, V, S-1, D)
              - pred_agent_slots: (B, T, V, D)
        """
        if start_slots.ndim != 3:
            raise ValueError(f"start_slots must be (B, V*S, D), got {tuple(start_slots.shape)}")
        if action_seq.ndim != 3:
            raise ValueError(f"action_seq must be (B, T, A), got {tuple(action_seq.shape)}")
        if start_slots.size(0) != action_seq.size(0):
            raise ValueError("Batch size mismatch between start_slots and action_seq.")

        current_slots = start_slots
        pred_slots = []
        pred_obj = []
        pred_agent = []
        for t in range(action_seq.size(1)):
            current_slots, obj_next, agent_next = self._predict_one_step(current_slots, action_seq[:, t, :])
            pred_slots.append(current_slots)
            pred_obj.append(obj_next)
            pred_agent.append(agent_next)

        if len(pred_slots) == 0:
            bsz, _, dim = start_slots.shape
            pred_slots_all = start_slots.new_empty((bsz, 0, self.num_views * self.slots_per_view, dim))
            pred_obj_all = start_slots.new_empty((bsz, 0, self.num_views, self.slots_per_view - 1, dim))
            pred_agent_all = start_slots.new_empty((bsz, 0, self.num_views, dim))
        else:
            pred_slots_all = torch.stack(pred_slots, dim=1)
            pred_obj_all = torch.stack(pred_obj, dim=1)
            pred_agent_all = torch.stack(pred_agent, dim=1)

        aux = {
            "pred_object_slots": pred_obj_all,
            "pred_agent_slots": pred_agent_all,
        }
        return pred_slots_all, aux

    def forward(
        self,
        history_slots: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            history_slots: (B, T_hist, V*S, D)
            future_actions: (B, T_pred, A)
        """
        if history_slots.ndim != 4:
            raise ValueError(
                f"history_slots must be (B, T_hist, V*S, D), got {tuple(history_slots.shape)}"
            )
        if future_actions.ndim != 3:
            raise ValueError(
                f"future_actions must be (B, T_pred, A), got {tuple(future_actions.shape)}"
            )
        if history_slots.size(0) != future_actions.size(0):
            raise ValueError("Batch size mismatch between history_slots and future_actions.")
        return self.rollout(history_slots[:, -1, :, :], future_actions)

    @torch.no_grad()
    def inference(self, history_slots: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        pred_future_slots, _ = self(history_slots, future_actions)
        return pred_future_slots
