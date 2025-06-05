import collections
import re
import time

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.episode_success = 0.0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        self.episode_success += info.get('success', 0.0)
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['success'] = float(self.episode_success > 0.0)
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                        self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class DMCEpisodeMonitor(EpisodeMonitor):
    def step(self, action):
        time_step = self.env.step(action)

        observation, reward = time_step.observation, time_step.reward
        terminated, truncated = time_step.last(), False
        info = dict(internal_state=time_step.physics)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()

        time_step = self.env.reset(*args, **kwargs)
        observation = time_step.observation
        info = dict(internal_state=time_step.physics)

        return observation, info


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.get_observation(), reward, terminated, truncated, info


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5,
                          reward_free=False, max_size=np.inf):
    """Make Offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        reward_free: Whether to use reward-free dataset.
        max_size: Maximum size of the dataset.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        from envs import ogbench_utils

        # use separate dataset for fine-tuning
        if not reward_free:
            splits = env_name.split('-')
            pos = splits.index('singletask')
            env_name = '-'.join(splits[: pos] + ['ft'] + splits[pos:])
        _, train_dataset, val_dataset = ogbench_utils.make_env_and_datasets(env_name)
        if train_dataset['observations'].shape[0] > max_size:
            for k, v in train_dataset.items():
                train_dataset[k] = v[:max_size]
        if val_dataset['observations'].shape[0] > max_size:
            for k, v in val_dataset.items():
                val_dataset[k] = v[:max_size]

        env = ogbench_utils.make_env_and_datasets(env_name, env_only=True)
        eval_env = ogbench_utils.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])

        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'walker' in env_name or 'cheetah' in env_name or 'quadruped' in env_name or 'jaco' in env_name:
        # ExORL
        from envs import dmc_utils

        env = dmc_utils.make_env(env_name)
        eval_env = dmc_utils.make_env(env_name)

        env = DMCEpisodeMonitor(env, filter_regexes=['internal_state'])
        eval_env = DMCEpisodeMonitor(eval_env, filter_regexes=['internal_state'])

        train_dataset, val_dataset = dmc_utils.get_dataset(
            env_name, reward_free=reward_free, max_size=max_size)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(
                actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(
                    actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
