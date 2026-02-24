import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from slotcontrast.modules import networks
from slotcontrast.utils import make_build_fn


@make_build_fn(__name__, "grouper")
def build(config, name: str):
    pass  # No special module building needed


def _to_grid_size(token_grid_size: Optional[Sequence[int]]) -> Optional[Tuple[int, int]]:
    if token_grid_size is None:
        return None
    if isinstance(token_grid_size, int):
        return (int(token_grid_size), int(token_grid_size))
    if len(token_grid_size) != 2:
        raise ValueError("token_grid_size must be [H, W] or a single int.")
    return (int(token_grid_size[0]), int(token_grid_size[1]))


def _infer_grid_size(n_tokens: int, token_grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if token_grid_size is not None:
        h, w = token_grid_size
        if h * w != n_tokens:
            raise ValueError(
                f"token_grid_size {token_grid_size} does not match token count {n_tokens}."
            )
        return h, w
    side = int(math.sqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(
            f"Can not infer square token grid from token count {n_tokens}. "
            "Set `token_grid_size` explicitly."
        )
    return side, side


def _build_token_coords(
    n_tokens: int,
    token_grid_size: Optional[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = _infer_grid_size(n_tokens, token_grid_size)
    y = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(n_tokens, 2)


def _compute_spatial_bias(
    prev_attn: torch.Tensor,
    token_coords: torch.Tensor,
    mode: str,
    sigma: float,
) -> torch.Tensor:
    """Compute (B, S, F) spatial bias from previous-step attention centroids."""
    if prev_attn.ndim != 3 or token_coords.ndim != 2 or token_coords.shape[1] != 2:
        raise ValueError("Expected prev_attn=(B,S,F) and token_coords=(F,2).")
    weights = prev_attn / (prev_attn.sum(dim=-1, keepdim=True) + 1e-8)
    centers = torch.einsum("bsf,fd->bsd", weights, token_coords)
    diff = token_coords.unsqueeze(0).unsqueeze(0) - centers.unsqueeze(2)
    dist2 = (diff * diff).sum(dim=-1)
    if mode == "gaussian":
        sigma2 = max(float(sigma) ** 2, 1e-8)
        return -dist2 / (2.0 * sigma2)
    if mode == "l2":
        return -dist2
    raise ValueError(f"Unknown spatial_bias_mode `{mode}`. Use 'gaussian' or 'l2'.")


class SlotAttention(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
        frozen: bool = False,
        spatial_bias_strength: float = 0.0,
        spatial_bias_sigma: float = 0.25,
        spatial_bias_mode: str = "gaussian",
        token_grid_size: Optional[Sequence[int]] = None,
        use_prev_attn_spatial_bias: bool = True,
    ):
        super().__init__()
        assert n_iters >= 1

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.to_k = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, kvq_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, kvq_dim, bias=False)

        if use_gru:
            self.gru = nn.GRUCell(input_size=kvq_dim, hidden_size=slot_dim)
        else:
            assert kvq_dim == slot_dim
            self.gru = None

        if hidden_dim is None:
            hidden_dim = 4 * slot_dim

        if use_mlp:
            self.mlp = networks.MLP(
                slot_dim, slot_dim, [hidden_dim], initial_layer_norm=True, residual=True
            )
        else:
            self.mlp = None

        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.n_iters = n_iters
        self.eps = eps
        self.scale = kvq_dim**-0.5
        self.spatial_bias_strength = float(spatial_bias_strength)
        self.spatial_bias_sigma = float(spatial_bias_sigma)
        self.spatial_bias_mode = str(spatial_bias_mode)
        if self.spatial_bias_mode not in ("gaussian", "l2"):
            raise ValueError("spatial_bias_mode must be 'gaussian' or 'l2'.")
        self.token_grid_size = _to_grid_size(token_grid_size)
        self.use_prev_attn_spatial_bias = bool(use_prev_attn_spatial_bias)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def step(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        prev_pre_norm_attn: Optional[torch.Tensor] = None,
        token_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one iteration of slot attention."""
        slots = self.norm_slots(slots)
        queries = self.to_q(slots)

        dots = torch.einsum("bsd, bfd -> bsf", queries, keys) * self.scale
        if (
            self.spatial_bias_strength > 0.0
            and self.use_prev_attn_spatial_bias
            and prev_pre_norm_attn is not None
            and token_coords is not None
        ):
            bias = _compute_spatial_bias(
                prev_pre_norm_attn, token_coords, self.spatial_bias_mode, self.spatial_bias_sigma
            )
            dots = dots + self.spatial_bias_strength * bias
        pre_norm_attn = torch.softmax(dots, dim=1)
        attn = pre_norm_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        if self.gru:
            updated_slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1))
            slots = updated_slots.unflatten(0, slots.shape[:2])
        else:
            slots = slots + updates

        if self.mlp is not None:
            slots = self.mlp(slots)

        return slots, pre_norm_attn

    def forward(self, slots: torch.Tensor, features: torch.Tensor, n_iters: Optional[int] = None):
        features = self.norm_features(features)
        keys = self.to_k(features)
        values = self.to_v(features)
        token_coords = None
        if self.spatial_bias_strength > 0.0 and self.use_prev_attn_spatial_bias:
            token_coords = _build_token_coords(
                features.shape[1], self.token_grid_size, features.device, features.dtype
            )

        if n_iters is None:
            n_iters = self.n_iters

        prev_pre_norm_attn = None
        for _ in range(n_iters):
            slots, pre_norm_attn = self.step(
                slots,
                keys,
                values,
                prev_pre_norm_attn=prev_pre_norm_attn,
                token_coords=token_coords,
            )
            prev_pre_norm_attn = pre_norm_attn

        return {"slots": slots, "masks": pre_norm_attn}

class GroupedSlotAttention(nn.Module):
    """Slot Attention with group-specific updates for BG/FG/AG slots.

    Notes:
      - Competition is slot-wise (softmax over slots for each feature/patch).
      - We optionally apply a different temperature ONLY to FG slots to sharpen FG competition.
      - Group-specific GRU/MLP can reduce negative transfer from heavily-supervised BG/AG slots to FG slots.
      - 'separate_group_softmax' is an experimental option: it removes cross-group competition
        (BG vs FG vs AG do NOT compete for the same features), which can stabilize groups but may
        also reduce decomposition sharpness unless paired with additional constraints.
    """

    def __init__(
        self,
        inp_dim: int,
        slot_dim: int,
        kvq_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        n_iters: int = 3,
        eps: float = 1e-8,
        use_gru: bool = True,
        use_mlp: bool = True,
        frozen: bool = False,
        num_bg_slots: int = 1,
        num_ag_slots: int = 1,
        num_fg_slots: Optional[int] = None,
        bg_slots: Optional[Sequence[int]] = None,
        fg_slots: Optional[Sequence[int]] = None,
        ag_slots: Optional[Sequence[int]] = None,
        grouped_update: bool = True,
        fg_temperature: float = 1.0,
        separate_group_softmax: bool = False,
        return_attn: bool = False,
        spatial_bias_strength: float = 0.0,
        spatial_bias_sigma: float = 0.25,
        spatial_bias_mode: str = "gaussian",
        token_grid_size: Optional[Sequence[int]] = None,
        use_prev_attn_spatial_bias: bool = True,
    ):
        super().__init__()
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")

        if kvq_dim is None:
            kvq_dim = slot_dim
        self.kvq_dim = int(kvq_dim)
        self.slot_dim = int(slot_dim)

        # Feature -> Key/Value, Slot -> Query
        self.to_k = nn.Linear(inp_dim, self.kvq_dim, bias=False)
        self.to_v = nn.Linear(inp_dim, self.kvq_dim, bias=False)
        self.to_q = nn.Linear(self.slot_dim, self.kvq_dim, bias=False)

        # Optional GRU update (slot-wise RNN refinement)
        if use_gru:
            if grouped_update:
                self.gru = None
                # Group-specific GRUs (BG/FG/AG)
                self.gru_bg = nn.GRUCell(input_size=self.kvq_dim, hidden_size=self.slot_dim)
                self.gru_fg = nn.GRUCell(input_size=self.kvq_dim, hidden_size=self.slot_dim)
                self.gru_ag = nn.GRUCell(input_size=self.kvq_dim, hidden_size=self.slot_dim)
            else:
                # Shared GRU (used only when grouped_update=False)
                self.gru = nn.GRUCell(input_size=self.kvq_dim, hidden_size=self.slot_dim)
                self.gru_bg = self.gru_fg = self.gru_ag = None
        else:
            # Without GRU, updates must match slot_dim to allow additive update
            if self.kvq_dim != self.slot_dim:
                raise ValueError("If use_gru=False, kvq_dim must equal slot_dim for additive updates.")
            self.gru = None
            self.gru_bg = self.gru_fg = self.gru_ag = None

        if hidden_dim is None:
            hidden_dim = 4 * self.slot_dim
        self.hidden_dim = int(hidden_dim)

        # Optional MLP refinement after GRU/additive update
        if use_mlp:
            if grouped_update:
                self.mlp = None
                self.mlp_bg = networks.MLP(
                    self.slot_dim, self.slot_dim, [self.hidden_dim], initial_layer_norm=True, residual=True
                )
                self.mlp_fg = networks.MLP(
                    self.slot_dim, self.slot_dim, [self.hidden_dim], initial_layer_norm=True, residual=True
                )
                self.mlp_ag = networks.MLP(
                    self.slot_dim, self.slot_dim, [self.hidden_dim], initial_layer_norm=True, residual=True
                )
            else:
                self.mlp = networks.MLP(
                    self.slot_dim, self.slot_dim, [self.hidden_dim], initial_layer_norm=True, residual=True
                )
                self.mlp_bg = self.mlp_fg = self.mlp_ag = None
        else:
            self.mlp = None
            self.mlp_bg = self.mlp_fg = self.mlp_ag = None

        # Normalization stabilizes training with strong frozen backbones (e.g., DINO features)
        self.norm_features = nn.LayerNorm(inp_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)

        self.n_iters = int(n_iters)
        self.eps = float(eps)
        self.scale = (self.kvq_dim) ** -0.5

        self.num_bg_slots = int(num_bg_slots)
        self.num_ag_slots = int(num_ag_slots)
        self.num_fg_slots = None if num_fg_slots is None else int(num_fg_slots)
        self.bg_slots = self._to_list(bg_slots)
        self.fg_slots = self._to_list(fg_slots)
        self.ag_slots = self._to_list(ag_slots)
        self.use_explicit = any(
            slots is not None for slots in (self.bg_slots, self.fg_slots, self.ag_slots)
        )

        self.grouped_update = bool(grouped_update)
        self.fg_temperature = float(fg_temperature)
        self.separate_group_softmax = bool(separate_group_softmax)
        self.return_attn = bool(return_attn)
        self.spatial_bias_strength = float(spatial_bias_strength)
        self.spatial_bias_sigma = float(spatial_bias_sigma)
        self.spatial_bias_mode = str(spatial_bias_mode)
        if self.spatial_bias_mode not in ("gaussian", "l2"):
            raise ValueError("spatial_bias_mode must be 'gaussian' or 'l2'.")
        self.token_grid_size = _to_grid_size(token_grid_size)
        self.use_prev_attn_spatial_bias = bool(use_prev_attn_spatial_bias)

        if frozen:
            for p in self.parameters():
                p.requires_grad = False

    @staticmethod
    def _to_list(slots: Optional[Sequence[int]]) -> Optional[List[int]]:
        if slots is None:
            return None
        if isinstance(slots, int):
            return [int(slots)]
        if isinstance(slots, (list, tuple)):
            return [int(s) for s in slots]
        if isinstance(slots, Sequence) and not isinstance(slots, (str, bytes)):
            return [int(s) for s in list(slots)]
        raise ValueError(f"Unsupported slot index type: {type(slots)}")

    @staticmethod
    def _resolve_indices(idxs: Sequence[int], n_slots: int) -> List[int]:
        out = []
        for idx in idxs:
            val = int(idx)
            if val < 0:
                val = n_slots + val
            if val < 0 or val >= n_slots:
                raise ValueError(f"Slot index {idx} out of range for K={n_slots}")
            out.append(val)
        return out

    @staticmethod
    def _is_empty(idx) -> bool:
        if isinstance(idx, slice):
            return idx.start == idx.stop
        return len(idx) == 0

    def _split_idxs(self, n_slots: int):
        """Split slot indices into BG / FG / AG slices."""
        if self.use_explicit:
            if self.bg_slots is None or self.fg_slots is None or self.ag_slots is None:
                raise ValueError("bg_slots, fg_slots, ag_slots must all be set for explicit indices.")
            bg = self._resolve_indices(self.bg_slots, n_slots)
            fg = self._resolve_indices(self.fg_slots, n_slots)
            ag = self._resolve_indices(self.ag_slots, n_slots)
            all_idx = bg + fg + ag
            if len(set(all_idx)) != len(all_idx):
                raise ValueError("Slot index lists overlap.")
            if len(all_idx) != n_slots:
                raise ValueError(
                    f"Slot index lists must cover all slots (K={n_slots}), got {len(all_idx)}."
                )
            return bg, fg, ag
        if self.num_bg_slots < 0 or self.num_ag_slots < 0:
            raise ValueError("num_bg_slots and num_ag_slots must be >= 0")

        fg_n = self.num_fg_slots
        if fg_n is None:
            fg_n = n_slots - self.num_bg_slots - self.num_ag_slots

        if fg_n < 0 or (self.num_bg_slots + fg_n + self.num_ag_slots != n_slots):
            raise ValueError(
                f"Slot split mismatch: K={n_slots}, bg={self.num_bg_slots}, fg={fg_n}, ag={self.num_ag_slots}"
            )

        bg = slice(0, self.num_bg_slots)
        fg = slice(self.num_bg_slots, self.num_bg_slots + fg_n)
        ag = slice(self.num_bg_slots + fg_n, n_slots)
        return bg, fg, ag

    def _group_softmax(self, dots: torch.Tensor, bg, fg, ag) -> torch.Tensor:
        """Compute slot-wise softmax within each group independently.

        WARNING: This removes cross-group competition (BG vs FG vs AG no longer compete
        for the same features). This may stabilize grouping but can also make attention
        'spreadier' unless additional constraints are introduced.
        """
        attn = torch.zeros_like(dots)
        for sl in (bg, fg, ag):
            if self._is_empty(sl):
                continue
            # softmax over slots dimension within the group slice
            attn[:, sl] = torch.softmax(dots[:, sl], dim=1)
        return attn

    def _update_group(self, slots: torch.Tensor, updates: torch.Tensor, gru: Optional[nn.GRUCell], mlp):
        """Apply GRU/additive update and optional MLP refinement to a slot group.

        IMPORTANT: Avoid unflatten pitfalls by explicit reshape/view with (B*S, D).
        """
        if gru is not None:
            b, s, d = slots.shape
            slots = gru(
                updates.reshape(b * s, -1),
                slots.reshape(b * s, -1),
            ).view(b, s, d)
        else:
            slots = slots + updates

        if mlp is not None:
            slots = mlp(slots)
        return slots

    def step(
        self,
        slots: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        prev_pre_attn: Optional[torch.Tensor] = None,
        token_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """One Slot Attention refinement step."""
        # Normalize slots and use the normalized slots for both attention and updates
        slots_norm = self.norm_slots(slots)
        q = self.to_q(slots_norm)  # (B, S, Dk)

        # dots: (B, S, F) where F is number of features/patches
        dots = torch.einsum("bsd, bfd -> bsf", q, keys) * self.scale
        if (
            self.spatial_bias_strength > 0.0
            and self.use_prev_attn_spatial_bias
            and prev_pre_attn is not None
            and token_coords is not None
        ):
            bias = _compute_spatial_bias(
                prev_pre_attn, token_coords, self.spatial_bias_mode, self.spatial_bias_sigma
            )
            dots = dots + self.spatial_bias_strength * bias

        bg, fg, ag = self._split_idxs(slots.shape[1])

        # Apply temperature ONLY to FG logits (sharpen FG competition).
        # Lower tau (<1) => more peaky assignments among FG slots.
        # NOTE: With global slot-wise softmax, sharpening FG can also let FG steal mass from BG/AG,
        # so consider balancing with group softmax or additional regularizers if needed.
        if self.fg_temperature != 1.0 and not self._is_empty(fg):
            dots[:, fg] = dots[:, fg] / self.fg_temperature

        # Slot-wise softmax over slots (competition across slots for each feature)
        if self.separate_group_softmax:
            pre_attn = self._group_softmax(dots, bg, fg, ag)
        else:
            pre_attn = torch.softmax(dots, dim=1)

        # Numerical stability: add epsilon then re-normalize over features for each slot.
        # This makes each slot distribute its attention mass over features.
        attn = pre_attn + self.eps
        attn = attn / attn.sum(-1, keepdim=True)

        # Compute per-slot updates as weighted sum of values
        updates = torch.einsum("bsf, bfd -> bsd", attn, values)

        # Use normalized slots as GRU/MLP state (matches baseline SlotAttention behavior)
        slots_for_update = slots_norm

        # Group-specific updates (BG/FG/AG) or shared updates
        if self.grouped_update:
            slots_out = slots_for_update.clone()
            if not self._is_empty(bg):
                slots_out[:, bg] = self._update_group(
                    slots_for_update[:, bg], updates[:, bg], self.gru_bg, self.mlp_bg
                )
            if not self._is_empty(fg):
                slots_out[:, fg] = self._update_group(
                    slots_for_update[:, fg], updates[:, fg], self.gru_fg, self.mlp_fg
                )
            if not self._is_empty(ag):
                slots_out[:, ag] = self._update_group(
                    slots_for_update[:, ag], updates[:, ag], self.gru_ag, self.mlp_ag
                )
            slots = slots_out
        else:
            slots = self._update_group(slots_for_update, updates, self.gru, self.mlp)

        # Return both the raw slot-softmax attention (pre_attn) and optionally the renormalized attention (attn)
        return slots, pre_attn, (attn if self.return_attn else None)

    def forward(
        self,
        slots: torch.Tensor,
        features: torch.Tensor,
        n_iters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forward pass.

        Args:
          slots: (B, S, slot_dim) initial slots
          features: (B, F, inp_dim) token/patch features
          n_iters: optional override for number of refinement iterations

        Returns:
          dict with:
            - "slots": (B, S, slot_dim)
            - "masks": (B, S, F) using pre_attn (slot-wise softmax outputs)
            - optionally "attn": (B, S, F) using renormalized attention (after eps+feature-norm)
        """
        # Normalize features before computing keys/values (helps with frozen feature backbones)
        features = self.norm_features(features)
        keys = self.to_k(features)    # (B, F, Dk)
        values = self.to_v(features)  # (B, F, Dk)

        if n_iters is None:
            n_iters = self.n_iters

        token_coords = None
        if self.spatial_bias_strength > 0.0 and self.use_prev_attn_spatial_bias:
            token_coords = _build_token_coords(
                features.shape[1], self.token_grid_size, features.device, features.dtype
            )

        attn = None
        pre_attn = None
        prev_pre_attn = None
        for _ in range(int(n_iters)):
            slots, pre_attn, attn = self.step(
                slots,
                keys,
                values,
                prev_pre_attn=prev_pre_attn,
                token_coords=token_coords,
            )
            prev_pre_attn = pre_attn

        out: Dict[str, Any] = {"slots": slots, "masks": pre_attn}
        if self.return_attn:
            out["attn"] = attn
        return out
