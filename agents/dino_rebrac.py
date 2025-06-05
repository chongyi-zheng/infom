import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, Param


class DINOReBRACAgent(flax.struct.PyTreeNode):
    """Self-distillation with no labels + revisited behavior-regularized actor-critic (DINO + ReBRAC) agent.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def dino_loss(target_repr, repr,
                  target_repr_center,
                  target_temp=0.04, temp=0.1):
        # stop_gradient
        target_repr = jax.lax.stop_gradient(target_repr)

        # center + sharpen
        logits = repr / temp
        target_probs = jax.nn.softmax((target_repr - target_repr_center) / target_temp, axis=-1)

        return -(target_probs * jax.nn.log_softmax(logits, axis=-1)).sum(axis=-1).mean()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the ReBRAC critic loss."""
        rng, sample_rng = jax.random.split(rng)
        next_reprs = self.network.select('target_encoder')(batch['next_observations'])
        next_dist = self.network.select('target_actor')(next_reprs)
        next_actions = next_dist.mode()
        noise = jnp.clip(
            (jax.random.normal(sample_rng, next_actions.shape) * self.config['actor_noise']),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)

        next_qs = self.network.select('target_critic')(next_reprs, actions=next_actions)
        next_q = next_qs.min(axis=0)

        mse = jnp.square(next_actions - batch['next_actions']).sum(axis=-1)
        next_q = next_q - self.config['alpha_critic'] * mse

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        reprs = self.network.select('encoder')(batch['observations'])
        q = self.network.select('critic')(reprs, actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def representation_loss(self, batch, grad_params, rng):
        """Compute the behavioral cloning loss for pretraining."""
        observations = batch['observations']

        rng, noise_rng1, noise_rng2 = jax.random.split(rng, 3)
        if self.config['encoder'] == 'mlp':
            noises1 = jnp.clip(
                (jax.random.normal(noise_rng1, observations.shape) * self.config['repr_noise']),
                -self.config['repr_noise_clip'],
                self.config['repr_noise_clip'],
            )
            noises2 = jnp.clip(
                (jax.random.normal(noise_rng2, observations.shape) * self.config['repr_noise']),
                -self.config['repr_noise_clip'],
                self.config['repr_noise_clip'],
            )
            aug1_observations = observations + noises1
            aug2_observations = observations + noises2
        elif 'impala' in self.config['encoder']:
            aug1_observations = batch['aug1_observations']
            aug2_observations = batch['aug2_observations']

        repr1 = self.network.select('encoder')(aug1_observations, params=grad_params)
        repr2 = self.network.select('encoder')(aug2_observations, params=grad_params)
        target_repr1 = self.network.select('target_encoder')(aug1_observations)
        target_repr2 = self.network.select('target_encoder')(aug2_observations)

        repr_loss = self.dino_loss(
            target_repr1, repr1, self.network.params['modules_target_repr_center']['value'],
            self.config['target_repr_temp'], self.config['repr_temp']
        ) / 2 + self.dino_loss(
            target_repr2, repr2, self.network.params['modules_target_repr_center']['value'],
            self.config['target_repr_temp'], self.config['repr_temp']
        ) / 2

        return repr_loss, {
            'repr_loss': repr_loss,
        }

    def behavioral_cloning_loss(self, batch, grad_params):
        """Compute the behavioral cloning loss for pretraining."""
        observations = batch['observations']
        actions = batch['actions']

        reprs = self.network.select('encoder')(observations)
        dist = self.network.select('actor')(reprs, params=grad_params)
        log_prob = dist.log_prob(actions)
        bc_loss = -log_prob.mean()

        return bc_loss, {
            'bc_loss': bc_loss,
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the ReBRAC actor loss."""
        reprs = self.network.select('encoder')(batch['observations'])
        dist = self.network.select('actor')(reprs, params=grad_params)
        actions = dist.mode()

        # Q loss.
        qs = self.network.select('critic')(reprs, actions=actions)
        q = jnp.min(qs, axis=0)

        # BC loss.
        mse = jnp.square(actions - batch['actions']).sum(axis=-1)

        # Normalize Q values by the absolute mean to make the loss scale invariant.
        actor_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            actor_loss = lam * actor_loss
        bc_loss = (self.config['alpha_actor'] * mse).mean()

        total_loss = actor_loss + bc_loss

        if self.config['tanh_squash']:
            action_std = dist._distribution.stddev()
        else:
            action_std = dist.stddev().mean()

        return total_loss, {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'std': action_std.mean(),
            'mse': mse.mean(),
        }

    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        """Compute the pre-training loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        repr_loss, repr_info = self.representation_loss(batch, grad_params, rng)
        for k, v in repr_info.items():
            info[f'repr/{k}'] = v

        bc_loss, bc_info = self.behavioral_cloning_loss(batch, grad_params)
        for k, v in bc_info.items():
            info[f'bc/{k}'] = v

        loss = repr_loss + bc_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetuning_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the fine-tuning loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = critic_loss + actor_loss
        return loss, info

    def target_reset(self):
        params = self.network.params
        params['modules_target_encoder'] = params['modules_encoder']

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def target_center_update(self, network, batch, rng):
        """
        Update center used for teacher output.
        """
        observations = batch['observations']

        rng, noise_rng1, noise_rng2 = jax.random.split(rng, 3)
        if self.config['encoder'] == 'mlp':
            noises1 = jnp.clip(
                (jax.random.normal(noise_rng1, observations.shape) * self.config['repr_noise']),
                -self.config['repr_noise_clip'],
                self.config['repr_noise_clip'],
            )
            noises2 = jnp.clip(
                (jax.random.normal(noise_rng2, observations.shape) * self.config['repr_noise']),
                -self.config['repr_noise_clip'],
                self.config['repr_noise_clip'],
            )
            aug1_observations = observations + noises1
            aug2_observations = observations + noises2
        elif 'impala' in self.config['encoder']:
            aug1_observations = batch['aug1_observations']
            aug2_observations = batch['aug2_observations']

        target_repr1 = self.network.select('target_encoder')(aug1_observations)
        target_repr2 = self.network.select('target_encoder')(aug2_observations)
        target_repr = jnp.stack([target_repr1, target_repr2], axis=0)
        repr_center = jnp.mean(target_repr, axis=0)

        # ema update
        target_repr_center = (repr_center * self.config['target_repr_center_tau'] +
                              network.params['modules_target_repr_center']['value'] * (1 - self.config['target_repr_center_tau']))
        network.params['modules_target_repr_center']['value'] = target_repr_center

    @jax.jit
    def pretrain(self, batch):
        """Pre-train the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.pretraining_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'encoder')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.finetuning_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if full_update:
            # Update the target networks only when `full_update` is True.
            self.target_update(new_network, 'encoder')
            self.target_update(new_network, 'critic')
            self.target_update(new_network, 'actor')
            new_rng, rng = jax.random.split(rng)
            self.target_center_update(new_network, batch, rng)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        reprs = self.network.select('encoder')(observations)
        dist = self.network.select('actor')(reprs, temperature=temperature)
        actions = dist.mode()
        noise = jnp.clip(
            (jax.random.normal(seed, actions.shape) * self.config['actor_noise'] * temperature),
            -self.config['actor_noise_clip'],
            self.config['actor_noise_clip'],
        )
        actions = jnp.clip(actions + noise, -1, 1)
        return actions

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
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_with_output_rng, init_rng = jax.random.split(rng, 3)

        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        assert config['encoder'] is not None
        encoder_module = encoder_modules[config['encoder']]
        encoders['encoder'] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            tanh_squash=config['tanh_squash'],
            state_dependent_std=False,
            const_std=True,
            final_fc_init_scale=config['actor_fc_scale'],
        )

        ex_reprs, _ = encoders.get('encoder').init_with_output(init_with_output_rng, ex_observations)
        target_repr_center_def = Param(shape=ex_reprs.shape[1:])

        network_info = dict(
            critic=(critic_def, (ex_reprs, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_reprs, ex_actions)),
            actor=(actor_def, (ex_reprs,)),
            target_actor=(copy.deepcopy(actor_def), (ex_reprs,)),
            # Add encoder to ModuleDict to make it separately callable.
            encoder=(encoders.get('encoder'), (ex_observations,)),
            target_encoder=(copy.deepcopy(encoders.get('encoder')), (ex_observations,)),
            target_repr_center=(target_repr_center_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = params['modules_actor']
        params['modules_target_encoder'] = params['modules_encoder']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dino_rebrac',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            tanh_squash=True,  # Whether to squash actions with tanh.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            alpha_actor=0.0,  # Actor BC coefficient.
            alpha_critic=0.0,  # Critic BC coefficient.
            actor_freq=4,  # Actor update frequency.
            normalize_q_loss=True,  # Whether to normalize the Q loss.
            actor_noise=0.2,  # Actor noise scale.
            actor_noise_clip=0.5,  # Actor noise clipping threshold.
            repr_noise=0.5,  # Representation noise scale.
            repr_noise_clip=1.0,  # Representation noise clipping threshold.
            target_repr_center_tau=0.1,  # Target representation center update rate
            repr_temp=0.1,  # Student representation temperature
            target_repr_temp=0.04,  # Teacher representation temperature
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name ('mlp', 'impala_small', etc.).
        )
    )
    return config
