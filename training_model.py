from __future__ import annotations

from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp


class TransformerBlock(eqx.Module):
    attention: eqx.nn.MultiheadAttention
    norm1: eqx.nn.LayerNorm
    ff_in: eqx.nn.Linear
    ff_out: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        qk_size: int,
        ff_dim: int,
        *,
        key: chex.PRNGKey,
    ):
        k1, k2, k3 = jax.random.split(key, 3)
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=model_dim,
            key_size=model_dim,
            value_size=model_dim,
            output_size=model_dim,
            qk_size=qk_size,
            vo_size=qk_size,
            key=k1,
        )
        self.norm1 = eqx.nn.LayerNorm(model_dim)
        self.ff_in = eqx.nn.Linear(model_dim, ff_dim, key=k2)
        self.ff_out = eqx.nn.Linear(ff_dim, model_dim, key=k3)
        self.norm2 = eqx.nn.LayerNorm(model_dim)

    def __call__(
        self,
        query: chex.Array,
        key_value: chex.Array,
        mask: chex.Array,
    ) -> chex.Array:
        attn_out = self.attention(query, key_value, key_value, mask=mask)
        h = jax.vmap(self.norm1)(query + attn_out)
        ff = jax.vmap(self.ff_out)(jax.nn.silu(jax.vmap(self.ff_in)(h)))
        return jax.vmap(self.norm2)(h + ff)


def _make_self_attention_mask(mask: chex.Array) -> chex.Array:
    base = jnp.logical_and(mask[:, None], mask[None, :])
    diagonal = jnp.eye(mask.shape[0], dtype=jnp.bool_)
    return jnp.logical_or(base, jnp.logical_and(jnp.logical_not(mask)[:, None], diagonal))


def _make_cross_attention_mask(mask: chex.Array) -> chex.Array:
    has_any = jnp.any(mask, axis=-1, keepdims=True)
    fallback = jnp.zeros_like(mask)
    fallback = fallback.at[:, 0].set(True)
    return jnp.where(has_any, mask, fallback)


def _infer_obs_structure(
    obs_dim: int,
    num_actions: int,
    obs_ems_feature_factor: int,
    obs_item_feature_factor: int,
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for num_ems in range(1, num_actions + 1):
        if num_actions % num_ems != 0:
            continue
        num_items = num_actions // num_ems
        if obs_ems_feature_factor * num_ems + obs_item_feature_factor * num_items + num_actions == obs_dim:
            candidates.append((num_ems, num_items))
    if not candidates:
        raise ValueError(
            f"Could not infer (obs_num_ems, max_num_items) from obs_dim={obs_dim}, num_actions={num_actions}."
        )
    candidates.sort(reverse=True)
    return candidates[0]


class PolicyTransformer(eqx.Module):
    obs_num_ems: int = eqx.field(static=True)
    max_num_items: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)

    obs_ems_feature_factor: int = eqx.field(static=True)
    obs_item_feature_factor: int = eqx.field(static=True)
    ems_coord_dim: int = eqx.field(static=True)
    item_feature_dim: int = eqx.field(static=True)
    mask_threshold: float = eqx.field(static=True)

    ems_projection: eqx.nn.Linear
    item_projection: eqx.nn.Linear
    self_ems_blocks: tuple[TransformerBlock, ...]
    self_item_blocks: tuple[TransformerBlock, ...]
    cross_ems_item_blocks: tuple[TransformerBlock, ...]
    cross_item_ems_blocks: tuple[TransformerBlock, ...]
    policy_ems_head: eqx.nn.Linear
    policy_item_head: eqx.nn.Linear
    flow_head: eqx.nn.Linear

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        *,
        obs_num_ems: int,
        max_num_items: int,
        key: chex.PRNGKey,
        num_layers: int,
        num_heads: int,
        ff_multiplier: int,
        qk_size_min: int,
        obs_ems_feature_factor: int,
        obs_item_feature_factor: int,
        ems_coord_dim: int,
        item_feature_dim: int,
        mask_threshold: float,
        key_count_base: int,
        key_count_per_layer: int,
        key_offset_initial: int,
        flow_input_multiplier: int,
        flow_output_dim: int,
        policy_ems_head_key_index: int,
        policy_item_head_key_index: int,
        flow_head_key_index: int,
    ):
        expected_obs_dim = (
            obs_ems_feature_factor * obs_num_ems
            + obs_item_feature_factor * max_num_items
            + (obs_num_ems * max_num_items)
        )
        if obs_dim != expected_obs_dim:
            raise ValueError(
                f"obs_dim mismatch: got {obs_dim}, expected {expected_obs_dim} for "
                f"obs_num_ems={obs_num_ems}, max_num_items={max_num_items}."
            )
        if num_actions != obs_num_ems * max_num_items:
            raise ValueError(
                f"num_actions mismatch: got {num_actions}, expected {obs_num_ems * max_num_items}."
            )

        self.obs_num_ems = obs_num_ems
        self.max_num_items = max_num_items
        self.num_actions = num_actions
        self.obs_ems_feature_factor = obs_ems_feature_factor
        self.obs_item_feature_factor = obs_item_feature_factor
        self.ems_coord_dim = ems_coord_dim
        self.item_feature_dim = item_feature_dim
        self.mask_threshold = mask_threshold

        qk_size = max(qk_size_min, hidden_dim // num_heads)
        ff_dim = ff_multiplier * hidden_dim

        num_keys = key_count_base + (key_count_per_layer * num_layers)
        keys = jax.random.split(key, num_keys)
        self.ems_projection = eqx.nn.Linear(ems_coord_dim, hidden_dim, key=keys[0])
        self.item_projection = eqx.nn.Linear(item_feature_dim, hidden_dim, key=keys[1])

        offset = key_offset_initial
        self.self_ems_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )
        offset += num_layers
        self.self_item_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )
        offset += num_layers
        self.cross_ems_item_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )
        offset += num_layers
        self.cross_item_ems_blocks = tuple(
            TransformerBlock(hidden_dim, num_heads, qk_size, ff_dim, key=keys[offset + i])
            for i in range(num_layers)
        )

        self.policy_ems_head = eqx.nn.Linear(
            hidden_dim,
            hidden_dim,
            key=keys[policy_ems_head_key_index],
        )
        self.policy_item_head = eqx.nn.Linear(
            hidden_dim,
            hidden_dim,
            key=keys[policy_item_head_key_index],
        )
        self.flow_head = eqx.nn.Linear(
            flow_input_multiplier * hidden_dim,
            flow_output_dim,
            key=keys[flow_head_key_index],
        )

    def _parse_obs(
        self, obs: chex.Array
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        num_ems = self.obs_num_ems
        num_items = self.max_num_items

        ems_coords_end = self.ems_coord_dim * num_ems
        ems_mask_end = ems_coords_end + num_ems
        item_feats_end = ems_mask_end + self.item_feature_dim * num_items
        item_mask_end = item_feats_end + num_items
        placed_end = item_mask_end + num_items
        action_mask_end = placed_end + (num_ems * num_items)

        ems_coords = obs[:ems_coords_end].reshape(num_ems, self.ems_coord_dim)
        ems_mask = obs[ems_coords_end:ems_mask_end] > self.mask_threshold
        item_feats = obs[ems_mask_end:item_feats_end].reshape(num_items, self.item_feature_dim)
        items_mask = obs[item_feats_end:item_mask_end] > self.mask_threshold
        items_placed = obs[item_mask_end:placed_end] > self.mask_threshold
        action_mask = obs[placed_end:action_mask_end].reshape(num_ems, num_items) > self.mask_threshold

        return ems_coords, ems_mask, item_feats, items_mask, items_placed, action_mask

    def __call__(self, obs: chex.Array) -> tuple[chex.Array, chex.Array]:
        ems_coords, ems_mask, item_feats, items_mask, items_placed, action_mask = self._parse_obs(obs)
        valid_items = jnp.logical_and(items_mask, jnp.logical_not(items_placed))

        ems_embeddings = jax.vmap(self.ems_projection)(ems_coords)
        item_embeddings = jax.vmap(self.item_projection)(item_feats)
        ems_embeddings = jnp.where(ems_mask[:, None], ems_embeddings, 0.0)
        item_embeddings = jnp.where(valid_items[:, None], item_embeddings, 0.0)

        ems_self_mask = _make_self_attention_mask(ems_mask)
        item_self_mask = _make_self_attention_mask(valid_items)
        action_mask = jnp.logical_and(action_mask, ems_mask[:, None])
        action_mask = jnp.logical_and(action_mask, valid_items[None, :])
        ems_cross_items_mask = _make_cross_attention_mask(action_mask)
        items_cross_ems_mask = _make_cross_attention_mask(jnp.swapaxes(action_mask, 0, 1))

        for self_ems, self_items, cross_ems_items, cross_items_ems in zip(
            self.self_ems_blocks,
            self.self_item_blocks,
            self.cross_ems_item_blocks,
            self.cross_item_ems_blocks,
        ):
            ems_embeddings = self_ems(ems_embeddings, ems_embeddings, ems_self_mask)
            item_embeddings = self_items(item_embeddings, item_embeddings, item_self_mask)
            new_ems_embeddings = cross_ems_items(
                ems_embeddings,
                item_embeddings,
                ems_cross_items_mask,
            )
            item_embeddings = cross_items_ems(
                item_embeddings,
                ems_embeddings,
                items_cross_ems_mask,
            )
            ems_embeddings = new_ems_embeddings

            ems_embeddings = jnp.where(ems_mask[:, None], ems_embeddings, 0.0)
            item_embeddings = jnp.where(valid_items[:, None], item_embeddings, 0.0)

        ems_policy = jax.vmap(self.policy_ems_head)(ems_embeddings)
        items_policy = jax.vmap(self.policy_item_head)(item_embeddings)
        logits_matrix = jnp.einsum("ek,ik->ei", ems_policy, items_policy)
        logits = logits_matrix.reshape(self.num_actions)

        ems_global = jnp.sum(jnp.where(ems_mask[:, None], ems_embeddings, 0.0), axis=0)
        items_global = jnp.sum(jnp.where(valid_items[:, None], item_embeddings, 0.0), axis=0)
        joint = jnp.concatenate([ems_global, items_global], axis=-1)
        log_flow = jnp.squeeze(self.flow_head(joint), axis=-1)
        return logits, log_flow


class PolicyMLP(PolicyTransformer):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_dim: int,
        *,
        key: chex.PRNGKey,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_multiplier: int = 2,
        qk_size_min: int = 1,
        obs_ems_feature_factor: int = 7,
        obs_item_feature_factor: int = 5,
        ems_coord_dim: int = 6,
        item_feature_dim: int = 3,
        mask_threshold: float = 0.5,
        key_count_base: int = 5,
        key_count_per_layer: int = 4,
        key_offset_initial: int = 2,
        flow_input_multiplier: int = 2,
        flow_output_dim: int = 1,
        policy_ems_head_key_index: int = -3,
        policy_item_head_key_index: int = -2,
        flow_head_key_index: int = -1,
    ):
        obs_num_ems, max_num_items = _infer_obs_structure(
            obs_dim,
            num_actions,
            obs_ems_feature_factor,
            obs_item_feature_factor,
        )
        super().__init__(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            obs_num_ems=obs_num_ems,
            max_num_items=max_num_items,
            key=key,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            qk_size_min=qk_size_min,
            obs_ems_feature_factor=obs_ems_feature_factor,
            obs_item_feature_factor=obs_item_feature_factor,
            ems_coord_dim=ems_coord_dim,
            item_feature_dim=item_feature_dim,
            mask_threshold=mask_threshold,
            key_count_base=key_count_base,
            key_count_per_layer=key_count_per_layer,
            key_offset_initial=key_offset_initial,
            flow_input_multiplier=flow_input_multiplier,
            flow_output_dim=flow_output_dim,
            policy_ems_head_key_index=policy_ems_head_key_index,
            policy_item_head_key_index=policy_item_head_key_index,
            flow_head_key_index=flow_head_key_index,
        )


def build_policy_transformer_from_config(
    *,
    obs_dim: int,
    num_actions: int,
    obs_num_ems: int,
    max_num_items: int,
    key: chex.PRNGKey,
    model_config: Any,
) -> PolicyTransformer:
    return PolicyTransformer(
        obs_dim=obs_dim,
        num_actions=num_actions,
        hidden_dim=model_config.hidden_dim,
        obs_num_ems=obs_num_ems,
        max_num_items=max_num_items,
        key=key,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        ff_multiplier=model_config.ff_multiplier,
        qk_size_min=model_config.qk_size_min,
        obs_ems_feature_factor=model_config.obs_ems_feature_factor,
        obs_item_feature_factor=model_config.obs_item_feature_factor,
        ems_coord_dim=model_config.ems_coord_dim,
        item_feature_dim=model_config.item_feature_dim,
        mask_threshold=model_config.mask_threshold,
        key_count_base=model_config.key_count_base,
        key_count_per_layer=model_config.key_count_per_layer,
        key_offset_initial=model_config.key_offset_initial,
        flow_input_multiplier=model_config.flow_input_multiplier,
        flow_output_dim=model_config.flow_output_dim,
        policy_ems_head_key_index=model_config.policy_ems_head_key_index,
        policy_item_head_key_index=model_config.policy_item_head_key_index,
        flow_head_key_index=model_config.flow_head_key_index,
    )
