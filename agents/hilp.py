import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCMetricValue, GCValue


class HILPAgent(flax.struct.PyTreeNode):
    """HILP agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['relabeled_rewards'] + self.config['discount'] * batch['relabeled_masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['relabeled_rewards'] + self.config['discount'] * batch['relabeled_masks'] * next_v1_t
        q2 = batch['relabeled_rewards'] + self.config['discount'] * batch['relabeled_masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def skill_value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_skill_critic')(
            batch['observations'], batch['skills'], batch['actions'], goal_encoded=True)
        q = jnp.minimum(q1, q2)
        v = self.network.select('skill_value')(
            batch['observations'], batch['skills'], goal_encoded=True, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def skill_critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('skill_value')(
            batch['next_observations'], batch['skills'], goal_encoded=True)

        q1, q2 = self.network.select('skill_critic')(
            batch['observations'], batch['skills'], batch['actions'],
            goal_encoded=True, params=grad_params
        )
        q = batch['skill_rewards'] + self.config['discount'] * batch['masks'] * next_v
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def skill_actor_loss(self, batch, grad_params):
        """Compute the actor loss."""
        v = self.network.select('skill_value')(
            batch['observations'], batch['skills'], goal_encoded=True)
        q1, q2 = self.network.select('skill_critic')(
            batch['observations'], batch['skills'], batch['actions'], goal_encoded=True)
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('skill_actor')(
            batch['observations'], batch['skills'], goal_encoded=True, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

        return actor_loss, actor_info

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, latent_rng = jax.random.split(rng)

        # Compute skill information
        batch['skills'], batch['skill_rewards'] = self.sample_latents(batch, latent_rng)

        # Train HILP representation.
        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        # Train skill policies.
        skill_value_loss, skill_value_info = self.skill_value_loss(batch, grad_params)
        for k, v in skill_value_info.items():
            info[f'skill_value/{k}'] = v

        skill_critic_loss, skill_critic_info = self.skill_critic_loss(batch, grad_params)
        for k, v in skill_critic_info.items():
            info[f'skill_critic/{k}'] = v

        skill_actor_loss, skill_actor_info = self.skill_actor_loss(batch, grad_params)
        for k, v in skill_actor_info.items():
            info[f'skill_actor/{k}'] = v

        loss = value_loss + skill_value_loss + skill_critic_loss + skill_actor_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetuning_loss(self, batch, grad_params, full_update=True, rng=None):
        info = {}

        batch['skills'] = batch['latents']
        batch['skill_rewards'] = batch['rewards']

        # Continue training skill policies.
        skill_value_loss, skill_value_info = self.skill_value_loss(batch, grad_params)
        for k, v in skill_value_info.items():
            info[f'skill_value/{k}'] = v

        skill_critic_loss, skill_critic_info = self.skill_critic_loss(batch, grad_params)
        for k, v in skill_critic_info.items():
            info[f'skill_critic/{k}'] = v

        if full_update:
            skill_actor_loss, skill_actor_info = self.skill_actor_loss(batch, grad_params)
            for k, v in skill_actor_info.items():
                info[f'skill_actor/{k}'] = v
        else:
            # Skip actor update.
            skill_actor_loss = 0.0

        loss = skill_value_loss + skill_critic_loss + skill_actor_loss
        return loss, info

    def target_reset(self):
        params = self.network.params
        params['modules_target_skill_critic'] = params['modules_skill_critic']

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def pretrain(self, batch):
        """Pre-train the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'skill_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.finetuning_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'skill_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        phis = self.get_phis(batch['next_observations']) - self.get_phis(batch['observations'])
        latent = jnp.linalg.lstsq(phis, batch['rewards'])[0]
        latent = latent / jnp.linalg.norm(latent) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        batch_size = batch['observations'].shape[0]
        latents = jax.random.normal(rng, shape=(batch_size, self.config['latent_dim']),
                                    dtype=batch['actions'].dtype)
        latents = latents / jnp.linalg.norm(latents, axis=1, keepdims=True) * jnp.sqrt(
            self.config['latent_dim'])

        phis = self.get_phis(batch['observations'])
        next_phis = self.get_phis(batch['next_observations'])
        rewards = ((next_phis - phis) * latents).sum(axis=1)

        return latents, rewards

    @jax.jit
    def sample_actions(
        self,
        observations,
        latents,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('skill_actor')(
            observations, latents, goal_encoded=True, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)

        return actions

    @jax.jit
    def get_phis(
            self,
            observations,
    ):
        """Return phi(s)."""
        _, phis, _ = self.network.select('value')(observations, observations, info=True)
        return phis[0]

    @classmethod
    def create(
            cls,
            seed,
            ex_observations,
            ex_actions,
            config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng, time_rng = jax.random.split(rng, 3)

        ex_latents = jnp.ones((ex_actions.shape[0], config['latent_dim']), dtype=ex_actions.dtype)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['skill_value'] = GCEncoder(state_encoder=encoder_module())
            encoders['skill_critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['skill_actor'] = GCEncoder(state_encoder=encoder_module())

        # Define value and actor networks.
        value_def = GCMetricValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            latent_dim=config['latent_dim'],
            num_ensembles=2,
            encoder=encoders.get('value'),
        )

        skill_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            gc_encoder=encoders.get('skill_value'),
        )
        skill_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('skill_critic'),
        )
        skill_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=ex_actions.shape[-1],
            state_dependent_std=False,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('skill_actor'),
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_observations)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_observations)),
            skill_critic=(skill_critic_def, (ex_observations, ex_latents, ex_actions, True)),
            target_skill_critic=(copy.deepcopy(skill_critic_def), (ex_observations, ex_latents, ex_actions, True)),
            skill_value=(skill_value_def, (ex_observations, ex_latents, None, True)),
            skill_actor=(skill_actor_def, (ex_observations, ex_latents, True)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_skill_critic'] = params['modules_skill_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='hilp',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # HILP latent dimension.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            actor_freq=4,  # Actor update frequency.
            alpha=10.0,  # AWR coefficient (need to be tuned for each environment).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            num_latent_inference_samples=10_000,  # Number of samples used to infer the task-specific latent.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name ('mlp', 'impala_small', etc.).
            # Dataset hyperparameters.
            relabel_reward=False,  # Whether to relabel the reward.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.625,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.375,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            value_geom_start_offset=1,  # Whether the support of the geometric sampling is [0, inf) or [1, inf)
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            actor_geom_start_offset=1,  # Whether the support the geometric sampling is [0, inf) or [1, inf)
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
        )
    )
    return config
