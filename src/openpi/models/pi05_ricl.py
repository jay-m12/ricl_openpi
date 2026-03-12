import dataclasses
import flax.nnx as nnx
from typing_extensions import override

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.model import (
    RiclObservationPi05,
    extract_observation_from_ricl_observation_pi05,
)
from openpi.models.pi0 import Pi0, make_attn_mask
import openpi.shared.array_typing as at

@dataclasses.dataclass(frozen=True)
class Pi05RiclConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"

    action_dim: int = 32
    action_horizon: int = 16
    max_token_len: int = 180

    num_retrieved_observations: int = 4

    pi05: bool = True

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05Ricl":
        return Pi05Ricl(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1):
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        obs_spec = _model.Observation(
            images={
                "base_0_rgb": image_spec,
                "left_wrist_0_rgb": image_spec,
                "right_wrist_0_rgb": image_spec,
            },
            image_masks={
                "base_0_rgb": image_mask_spec,
                "left_wrist_0_rgb": image_mask_spec,
                "right_wrist_0_rgb": image_mask_spec,
            },
            state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
            tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
        )

        ricl_obs_spec = _model.RiclObservationPi05(
            query_images=obs_spec.images,
            query_image_masks=obs_spec.image_masks,
            query_state=obs_spec.state,
            query_tokenized_prompt=obs_spec.tokenized_prompt,
            query_tokenized_prompt_mask=obs_spec.tokenized_prompt_mask,

            retrieved_0_images=obs_spec.images,
            retrieved_0_image_masks=obs_spec.image_masks,
            retrieved_0_state=obs_spec.state,
            retrieved_0_tokenized_prompt=obs_spec.tokenized_prompt,
            retrieved_0_tokenized_prompt_mask=obs_spec.tokenized_prompt_mask,

            retrieved_1_images=obs_spec.images,
            retrieved_1_image_masks=obs_spec.image_masks,
            retrieved_1_state=obs_spec.state,
            retrieved_1_tokenized_prompt=obs_spec.tokenized_prompt,
            retrieved_1_tokenized_prompt_mask=obs_spec.tokenized_prompt_mask,

            retrieved_2_images=obs_spec.images,
            retrieved_2_image_masks=obs_spec.image_masks,
            retrieved_2_state=obs_spec.state,
            retrieved_2_tokenized_prompt=obs_spec.tokenized_prompt,
            retrieved_2_tokenized_prompt_mask=obs_spec.tokenized_prompt_mask,

            retrieved_3_images=obs_spec.images,
            retrieved_3_image_masks=obs_spec.image_masks,
            retrieved_3_state=obs_spec.state,
            retrieved_3_tokenized_prompt=obs_spec.tokenized_prompt,
            retrieved_3_tokenized_prompt_mask=obs_spec.tokenized_prompt_mask,

            exp_lamda_distances=jax.ShapeDtypeStruct(
                [batch_size, self.num_retrieved_observations + 1, 1], jnp.float32
            ),
        )

        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )

        return ricl_obs_spec, action_spec

    @override
    def fake_obs(self, batch_size: int = 1) -> _model.Observation:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        obs_spec = _model.Observation(
            images={
                "base_0_rgb": image_spec,
                "left_wrist_0_rgb": image_spec,
                "right_wrist_0_rgb": image_spec,
            },
            image_masks={
                "base_0_rgb": image_mask_spec,
                "left_wrist_0_rgb": image_mask_spec,
                "right_wrist_0_rgb": image_mask_spec,
            },
            state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
            tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
            tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_),
        )

        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs_spec)

class Pi05Ricl(Pi0):
    def __init__(self, config: Pi05RiclConfig, rngs: nnx.Rngs):
        self.config = config
        super().__init__(config, rngs)
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: RiclObservationPi05,
        actions: _model.Actions,
        *,
        train: bool = False,
    ):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ricl(
            observation,
            num_retrieved_observations=self.config.num_retrieved_observations,
            train=train,
        )

        query_obs = extract_observation_from_ricl_observation_pi05(
            observation, "query_"
        )
        query_obs = _model.preprocess_observation(preprocess_rng, query_obs, train=train)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            query_obs, x_t, time
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t))

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: RiclObservationPi05,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ):
        query_obs = extract_observation_from_ricl_observation_pi05(
            observation, "query_"
        )
        query_obs = _model.preprocess_observation(None, query_obs, train=False)

        dt = -1.0 / num_steps
        batch_size = query_obs.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_ricl(
            observation,
            num_retrieved_observations=self.config.num_retrieved_observations,
            train=False,
        )

        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                query_obs, x_t, jnp.broadcast_to(time, batch_size)
            )

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_attn_mask = jnp.repeat(
                prefix_mask[:, None, :],
                suffix_tokens.shape[1],
                axis=1,
            )
            full_attn_mask = jnp.concatenate([prefix_to_suffix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def embed_prefix_single(
        self,
        observation: _model.Observation,
        *,
        train: bool = False,
    ):
        observation = _model.preprocess_observation(None, observation, train=train)
        return super().embed_prefix(observation)

    def embed_prefix_ricl(
        self,
        ricl_observation: RiclObservationPi05,
        *,
        num_retrieved_observations: int,
        train: bool = False,
    ):
        list_of_prefix_tokens = []
        list_of_prefix_masks = []
        list_of_prefix_ar_masks = []

        for i in range(num_retrieved_observations):
            obs = extract_observation_from_ricl_observation_pi05(
                ricl_observation, f"retrieved_{i}_"
            )
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix_single(
                obs, train=train
            )
            list_of_prefix_tokens.append(prefix_tokens)
            list_of_prefix_masks.append(prefix_mask)
            list_of_prefix_ar_masks.append(prefix_ar_mask)

        query_obs = extract_observation_from_ricl_observation_pi05(
            ricl_observation, "query_"
        )
        query_prefix_tokens, query_prefix_mask, query_prefix_ar_mask = self.embed_prefix_single(
            query_obs, train=train
        )
        list_of_prefix_tokens.append(query_prefix_tokens)
        list_of_prefix_masks.append(query_prefix_mask)
        list_of_prefix_ar_masks.append(query_prefix_ar_mask)

        prefix_tokens = jnp.concatenate(list_of_prefix_tokens, axis=1)
        prefix_mask = jnp.concatenate(list_of_prefix_masks, axis=1)
        prefix_ar_mask = jnp.concatenate(list_of_prefix_ar_masks, axis=0)

        return prefix_tokens, prefix_mask, prefix_ar_mask


