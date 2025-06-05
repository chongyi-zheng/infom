from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class Param(nn.Module):
    """Scalar parameter module."""

    shape: Sequence[int] = ()
    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full(self.shape, self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def __getattr__(self, name):
        return getattr(self.distribution, name)

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        value_net = mlp_class(hidden_dims=(*self.hidden_dims, self.value_dim),
                              activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.value_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        else:
            v = self.value_net(inputs)

        return v


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class IntentionEncoder(nn.Module):
    """Transition encoder network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self):
        self.trunk_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.latent_dim, kernel_init=default_init())
        self.log_std_net = nn.Dense(self.latent_dim, kernel_init=default_init())

    def __call__(
        self,
        observations,
        actions,
    ):
        """Return latent distribution.

        Args:
            observations: Observations.
            actions: Actions.
        """
        if self.encoder is not None:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.trunk_net(inputs)

        means = self.mean_net(outputs)
        log_stds = self.log_std_net(outputs)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))

        return distribution


class VectorField(nn.Module):
    """Flow-matching vector field function.

    This module can be used for both value velocity field u(s, g) and critic velocity filed u(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: state encoder.
    """

    vector_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)

        self.vector_field_net = mlp_class(
            hidden_dims=(*self.hidden_dims, self.vector_dim),
            activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, noisy_goals, times, observations=None, actions=None, latents=None):
        """Return the value/critic velocity field.

        Args:
            noisy_goals: Noisy goals.
            times: Times.
            observations: Observations (Optional).
            actions: Actions (Optional).
        """
        if self.encoder is not None:
            noisy_goals = self.encoder(noisy_goals)

        if self.encoder is not None and observations is not None:
            observations = self.encoder(observations)

        times = times[..., None]
        inputs = [noisy_goals, times]
        if observations is not None:
            inputs.append(observations)
        if actions is not None:
            inputs.append(actions)
        if latents is not None:
            inputs.append(latents)
        inputs = jnp.concatenate(inputs, axis=-1)

        vf = self.vector_field_net(inputs)

        return vf


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm
        )

        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded (optional).
            temperature: Scaling factor for the standard deviation (optional).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        value_dim: Value dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    value_dim: int = 1
    num_residual_blocks: int = 1
    layer_norm: bool = True
    num_ensembles: int = 1
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.value_net = mlp_class(
            (*self.hidden_dims, self.value_dim),
            activate_final=False,
            layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals=None, actions=None, goal_encoded=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
            goal_encoded: Whether the goals are already encoded (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals, goal_encoded=goal_encoded)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        if self.value_dim == 1:
            v = self.value_net(inputs).squeeze(-1)
        else:
            v = self.value_net(inputs)

        return v


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    num_ensembles: int = 1
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_class = MLP

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class,  self.num_ensembles)

        self.phi = mlp_class(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False, layer_norm=self.layer_norm
        )
        self.psi = mlp_class(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False, layer_norm=self.layer_norm
        )

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        if len(phi.shape) == 2:  # Non-ensemble.
            v = jnp.einsum('ik,jk->ij', phi, psi) / jnp.sqrt(self.latent_dim)
        else:
            v = jnp.einsum('eik,ejk->eij', phi, psi) / jnp.sqrt(self.latent_dim)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCMetricValue(nn.Module):
    """Metric value function.

    This module computes the value function as || \phi(s) - \phi(g) ||_2.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    num_ensembles: int = 1
    encoder: nn.Module = None

    def setup(self) -> None:
        network_module = MLP
        if self.num_ensembles > 1:
            network_module = ensemblize(network_module,  self.num_ensembles)
        self.phi = network_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the metric value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        squared_dist = ((phi_s - phi_g) ** 2).sum(axis=-1)
        v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-12))

        if info:
            return v, phi_s, phi_g
        else:
            return v
