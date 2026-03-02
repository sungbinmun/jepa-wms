import torch
from torch import nn


class AgentCausalSlotPredictor(nn.Module):
    """Agent-centric slot predictor with causal structural bias.

    Assumptions:
    - Input slot order is fixed.
    - The last slot is the agent slot.
    - Action at step t should only drive the update for the agent slot t -> t+1.
    - Non-agent slots are updated conditioned on non-agent(t), agent(t), and
      stop-gradient(agent(t+1)).
    """

    def __init__(
        self,
        slot_dim: int,
        action_dim: int,
        num_heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        stop_gradient_agent_to_object: bool = True,
    ):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.action_dim = int(action_dim)
        self.stop_gradient_agent_to_object = bool(stop_gradient_agent_to_object)

        # Agent transition: query from (agent_t, action_t), attend over all slots_t.
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

        # Object transition: query from non-agent_t, attend over
        # [non-agent_t, agent_t, stopgrad(agent_t+1)].
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

    def _predict_one_step(
        self,
        slots_t: torch.Tensor,
        action_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            slots_t: (B, S, D), with agent slot at index -1.
            action_t: (B, A), action for transition t -> t+1.
        Returns:
            slots_next: (B, S, D)
            non_agent_next: (B, S-1, D)
            agent_next: (B, 1, D)
        """
        if slots_t.ndim != 3:
            raise ValueError(f"slots_t must be (B, S, D), got {tuple(slots_t.shape)}")
        if action_t.ndim != 2:
            raise ValueError(f"action_t must be (B, A), got {tuple(action_t.shape)}")
        if slots_t.size(1) < 2:
            raise ValueError("Need at least 2 slots (one non-agent + one agent).")

        non_agent_t = slots_t[:, :-1, :]  # (B, S-1, D)
        agent_t = slots_t[:, -1:, :]  # (B, 1, D)

        action_embed = self.action_proj(action_t).unsqueeze(1)  # (B, 1, D)
        agent_query = self.agent_query_proj(torch.cat([agent_t, action_embed], dim=-1))

        q_agent = self.agent_q_norm(agent_query)
        kv_agent = self.agent_kv_norm(slots_t)
        delta_agent, _ = self.agent_cross_attn(q_agent, kv_agent, kv_agent, need_weights=False)
        agent_next = agent_t + delta_agent
        agent_next = agent_next + self.agent_ff(self.agent_ff_norm(agent_next))

        if self.stop_gradient_agent_to_object:
            agent_t_for_objects = agent_t.detach()
            agent_next_for_objects = agent_next.detach()
        else:
            agent_t_for_objects = agent_t
            agent_next_for_objects = agent_next
        obj_context = torch.cat([non_agent_t, agent_t_for_objects, agent_next_for_objects], dim=1)

        q_obj = self.obj_q_norm(non_agent_t)
        kv_obj = self.obj_kv_norm(obj_context)
        delta_obj, _ = self.obj_cross_attn(q_obj, kv_obj, kv_obj, need_weights=False)
        non_agent_next = non_agent_t + delta_obj
        non_agent_next = non_agent_next + self.obj_ff(self.obj_ff_norm(non_agent_next))

        slots_next = torch.cat([non_agent_next, agent_next], dim=1)
        return slots_next, non_agent_next, agent_next

    def forward(
        self,
        history_slots: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            history_slots: (B, T_hist, S, D)
            future_actions: (B, T_pred, A), action[t] for each predicted transition
        Returns:
            pred_future_slots: (B, T_pred, S, D)
            aux: dict with per-branch predictions
        """
        if history_slots.ndim != 4:
            raise ValueError(
                f"history_slots must be (B, T_hist, S, D), got {tuple(history_slots.shape)}"
            )
        if future_actions.ndim != 3:
            raise ValueError(
                f"future_actions must be (B, T_pred, A), got {tuple(future_actions.shape)}"
            )
        if history_slots.size(0) != future_actions.size(0):
            raise ValueError("Batch size mismatch between history_slots and future_actions.")

        current_slots = history_slots[:, -1, :, :]  # use latest observed slots as transition source
        pred_all = []
        pred_obj = []
        pred_agent = []
        for t in range(future_actions.size(1)):
            current_slots, obj_next, agent_next = self._predict_one_step(current_slots, future_actions[:, t, :])
            pred_all.append(current_slots)
            pred_obj.append(obj_next)
            pred_agent.append(agent_next)

        pred_future_slots = torch.stack(pred_all, dim=1)  # (B, T_pred, S, D)
        pred_object_slots = torch.stack(pred_obj, dim=1)  # (B, T_pred, S-1, D)
        pred_agent_slots = torch.stack(pred_agent, dim=1).squeeze(2)  # (B, T_pred, D)
        aux = {
            "pred_object_slots": pred_object_slots,
            "pred_agent_slots": pred_agent_slots,
        }
        return pred_future_slots, aux

    @torch.no_grad()
    def inference(self, history_slots: torch.Tensor, future_actions: torch.Tensor) -> torch.Tensor:
        pred_future_slots, _ = self(history_slots, future_actions)
        return pred_future_slots
