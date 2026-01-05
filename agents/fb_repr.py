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
from utils.networks import GCActor, GCValue, FBForwardMLP, FBBackwardMLP


class ForwardBackwardRepresentationAgent(flax.struct.PyTreeNode):
    """Forward Backward Representation + Implicit Q-Learning (FB + IQL) agent.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        q1, q2 = self.network.select('target_critic')(
            batch['observations'], batch['latents'], batch['actions'], goal_encoded=True)
        q = jnp.minimum(q1, q2)
        v = self.network.select('value')(
            batch['observations'], batch['latents'], goal_encoded=True, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['expectile']).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(
            batch['next_observations'], batch['latents'], goal_encoded=True)

        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['latents'], batch['actions'], goal_encoded=True,
            params=grad_params
        )
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params):
        """Compute the IQL actor loss."""
        v = self.network.select('value')(
            batch['observations'], batch['latents'], goal_encoded=True)
        q1, q2 = self.network.select('critic')(
            batch['observations'], batch['latents'], batch['actions'], goal_encoded=True)
        q = jnp.minimum(q1, q2)
        adv = q - v

        exp_a = jnp.exp(adv * self.config['alpha_awr'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(
            batch['observations'], batch['latents'], goal_encoded=True, params=grad_params)
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
    def forward_backward_repr_loss(self, batch, grad_params, rng):
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']
        latents = batch['latents']

        next_dist = self.network.select('actor')(
            next_observations, latents, goal_encoded=True, temperature=1.0)
        next_actions = jnp.clip(next_dist.mode(), -1, 1)
        next_forward_reprs = self.network.select('target_forward_repr')(
            next_observations, latents, next_actions, goal_encoded=True)
        next_backward_reprs = self.network.select('target_backward_repr')(
            next_observations)
        next_backward_reprs = next_backward_reprs / jnp.linalg.norm(
            next_backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        target_occ_measures = jnp.einsum('esd,td->est', next_forward_reprs, next_backward_reprs)
        if self.config['repr_agg'] == 'mean':
            target_occ_measures = jnp.mean(target_occ_measures, axis=0)
        else:
            target_occ_measures = jnp.min(target_occ_measures, axis=0)

        forward_reprs = self.network.select('forward_repr')(
            observations, latents, actions, goal_encoded=True, params=grad_params)
        backward_reprs = self.network.select('backward_repr')(
            next_observations, params=grad_params)
        backward_reprs = backward_reprs / jnp.linalg.norm(
            backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        occ_measures = jnp.einsum('esd,td->est', forward_reprs, backward_reprs)

        err = occ_measures - self.config['discount'] * target_occ_measures[None]
        B = err.shape[-1]
        I = jnp.eye(B)
        mask = 1 - I
        off_count = B * (B - 1)

        repr_off_diag_loss = 0.5 * jnp.sum(
            jax.vmap(lambda x: jnp.sum((x * (1 - I)) ** 2) / off_count, in_axes=0)(err)
        )
        repr_diag_loss = -jnp.sum(jax.vmap(lambda x: jnp.mean(jnp.diag(x)), in_axes=0)(occ_measures))
        repr_loss = repr_off_diag_loss + repr_diag_loss
        
        # orthonormalization loss
        covariance = jnp.matmul(backward_reprs, backward_reprs.T)
        ortho_diag_loss = -2 * jnp.diag(covariance).mean()
        ortho_off_diag_loss = ((covariance * mask) ** 2).sum() / off_count 
        ortho_loss = ortho_diag_loss + ortho_off_diag_loss
        fb_loss = repr_loss + self.config['orthonorm_coeff'] * ortho_loss

        return fb_loss, {
            'target_M': target_occ_measures.mean(),
            'M1': occ_measures[0].mean(),
            'F1': forward_reprs[0].mean(),
            'B': backward_reprs.mean(),
            'B_norm': jnp.linalg.norm(backward_reprs, axis=-1).mean(),
            'fb_loss': fb_loss,
            'fb_diag': repr_diag_loss,
            'fb_offdiag': repr_off_diag_loss,
            'orth_loss': ortho_loss,
            'orth_loss_diag': ortho_diag_loss,
            'orth_loss_offdiag': ortho_off_diag_loss
        }

    def forward_backward_actor_loss(self, batch, grad_params, rng=None):
        observations = batch['observations']
        actions = batch['actions']
        latents = batch['latents']

        dist = self.network.select('actor')(
            observations, latents, goal_encoded=True, params=grad_params, temperature=1.0)
        q_actions = jnp.clip(dist.mode(), -1, 1)
        forward_reprs = self.network.select('forward_repr')(
            observations, latents, q_actions, goal_encoded=True)
        q1, q2 = jnp.einsum('esd,sd->es', forward_reprs, latents)
        q = jnp.minimum(q1, q2)
        q_loss = -q.mean()
        
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss
            
        actor_loss = q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'q_loss': q_loss,
            'q_mean': q.mean(),
            'q_abs_mean': jnp.abs(q).mean(),
            'mse': jnp.mean((dist.mode() - actions) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }
    
    @jax.jit
    def pretraining_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, latent_rng, fb_repr_rng, actor_rng = jax.random.split(rng, 4)

        # sample latents and mix
        batch['latents'] = self.sample_latents(batch, latent_rng)

        fb_repr_loss, fb_repr_info = self.forward_backward_repr_loss(batch, grad_params, fb_repr_rng)
        for k, v in fb_repr_info.items():
            info[f'fb_repr/{k}'] = v

        fb_actor_loss, fb_actor_info = self.forward_backward_actor_loss(batch, grad_params, actor_rng)
        for k, v in fb_actor_info.items():
            info[f'fb_actor/{k}'] = v

        loss = fb_repr_loss + fb_actor_loss
        return loss, info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetuning_loss(self, batch, grad_params, full_update=True, rng=None):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        if full_update:
            # Update the actor.
            actor_loss, actor_info = self.actor_loss(batch, grad_params)
            for k, v in actor_info.items():
                info[f'actor/{k}'] = v
        else:
            # Skip actor update.
            actor_loss = 0.0

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree.map(
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
        self.target_update(new_network, 'forward_repr')
        self.target_update(new_network, 'backward_repr')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('full_update',))
    def finetune(self, batch, full_update=True):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.finetuning_loss(batch, grad_params, full_update, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def infer_latent(self, batch):
        backward_reprs = self.network.select('backward_repr')(batch['observations'])
        backward_reprs = backward_reprs / jnp.linalg.norm(
            backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        # reward-weighted average
        latent = jnp.matmul(batch['rewards'].T, backward_reprs) / batch['rewards'].shape[0]
        latent = latent / jnp.linalg.norm(latent) * jnp.sqrt(self.config['latent_dim'])

        return latent

    @jax.jit
    def sample_latents(self, batch, rng):
        latent_rng, perm_rng, mix_rng = jax.random.split(rng, 3)

        batch_size = batch['observations'].shape[0]
        latents = jax.random.normal(latent_rng, shape=(batch_size, self.config['latent_dim']),
                                    dtype=batch['actions'].dtype)
        latents = latents / jnp.linalg.norm(
            latents, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        perm = jax.random.permutation(perm_rng, jnp.arange(batch_size))
        backward_reprs = self.network.select('backward_repr')(batch['observations'][perm])
        backward_reprs = backward_reprs / jnp.linalg.norm(
            backward_reprs, axis=-1, keepdims=True) * jnp.sqrt(self.config['latent_dim'])
        backward_reprs = jax.lax.stop_gradient(backward_reprs)
        latents = jnp.where(
            jax.random.uniform(mix_rng, (batch_size, 1)) < self.config['latent_mix_prob'],
            latents,
            backward_reprs
        )

        return latents

    @jax.jit
    def sample_actions(
        self,
        observations,
        latents,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, latents,
                                            goal_encoded=True, temperature=temperature)
        actions = jnp.clip(dist.mode(), -1, 1)
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
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        ex_latents = jnp.ones((*ex_actions.shape[:-1], config['latent_dim']),
                            dtype=ex_actions.dtype)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(state_encoder=encoder_module())
            encoders['critic'] = GCEncoder(state_encoder=encoder_module())
            encoders['forward_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['backward_repr'] = GCEncoder(state_encoder=encoder_module())
            encoders['actor'] = GCEncoder(state_encoder=encoder_module())

        # Define networks.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('value'),
        )
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('critic'),
        )
        forward_repr_def = FBForwardMLP(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            layer_norm=config['fb_repr_layer_norm'],
            num_ensembles=2,
            gc_encoder=encoders.get('forward_repr'),
        )
        backward_repr_def = FBBackwardMLP(
            hidden_dims=config['forward_repr_hidden_dims'],
            value_dim=config['latent_dim'],
            layer_norm=config['fb_repr_layer_norm'],
            num_ensembles=1,
            gc_encoder=encoders.get('backward_repr'),
        )
        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            tanh_squash=True,
            layer_norm=config['actor_layer_norm'],
            const_std=config['const_std'],
            gc_encoder=encoders.get('actor'),
        )

        network_info = dict(
            value=(value_def, (ex_observations, ex_latents, None, True)),
            critic=(critic_def, (ex_observations, ex_latents, ex_actions, True)),
            forward_repr=(forward_repr_def, (ex_observations, ex_latents, ex_actions, True)),
            backward_repr=(backward_repr_def, (ex_observations, )),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_latents, ex_actions, True)),
            target_forward_repr=(copy.deepcopy(forward_repr_def), (ex_observations, ex_latents, ex_actions, True)),
            target_backward_repr=(copy.deepcopy(backward_repr_def), (ex_observations, )),
            actor=(actor_def, (ex_observations, ex_latents, True)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=config['lr'])
        )
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_forward_repr'] = params['modules_forward_repr']
        params['modules_target_backward_repr'] = params['modules_backward_repr']
        network = network.replace(params=params)
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fb_repr',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=512,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            forward_repr_hidden_dims=(1024, 1024, 1024),  # Forward representation network hidden dimensions.
            backward_repr_hidden_dims=(256, 256, 256),  # Backward representation network hidden dimension.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=False,  # Whether to use layer normalization for the critic.
            fb_repr_layer_norm=True,  # Whether to use layer normalization for the forward and the backward representations.
            latent_dim=50,  # Latent dimension for transition latents.
            discount=0.98,  # Discount factor.
            tau=0.01,  # Target network update rate.
            expectile=0.9,  # IQL style expectile.
            actor_freq=4,  # Actor update frequency.
            repr_agg='mean',  # Aggregation method for target forward backward representation.
            orthonorm_coeff=1.0,  # orthonormalization coefficient
            latent_mix_prob=0.5,  # Probability to replace latents sampled from gaussian with backward representations.
            alpha_repr=1.0,  # Temperature in BC coefficient in DDPG+BC.
            alpha_awr=1.0,  # Temperature in IQL.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            num_latent_inference_samples=10_000,  # Number of samples used to infer the task-specific latent.
            encoder=ml_collections.config_dict.placeholder(str),  # Encoder name ('mlp', 'impala_small', etc.).
        )
    )
    return config
