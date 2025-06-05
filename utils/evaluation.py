from collections import defaultdict
import xml.etree.ElementTree as ET

import jax
import numpy as np
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    dataset=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    inferred_latent=None,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        dataset: Dataset.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        inferred_latent: Latent to be used for evaluation (only for HILP and FB).
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        if dataset is not None:
            observation = dataset.normalize_observations(observation)
        done = False
        step = 0
        render = []
        while not done:
            if inferred_latent is not None:
                action = actor_fn(observations=observation, latents=inferred_latent, temperature=eval_temperature)
            else:
                action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            if dataset is not None:
                next_observation = dataset.normalize_observations(next_observation)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_gc(
    agent,
    env,
    task_id=None,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
        goal = info.get('goal')
        goal_frame = info.get('goal_rendered')
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
            action = np.array(action)
            if not config.get('discrete'):
                if eval_gaussian is not None:
                    action = np.random.normal(action, eval_gaussian)
                action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                if goal_frame is not None:
                    render.append(np.concatenate([goal_frame, frame], axis=0))
                else:
                    render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_octo(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment. 
    Adapt from https://colab.research.google.com/github/simpler-env/SimplerEnv/blob/main/example.ipynb.

    Args:
        agent: Agent.
        env: Environment.
        dataset: Dataset.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        while not done:
            action = actor_fn(
                observations=observation,
                tasks=env.task,
                temperature=eval_temperature,
            )
            action = np.array(action)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                frame = observation['image_primary'][0, -1]
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation
        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders


def evaluate_policy_evaluation(
    estimator,
    dataset,
    num_eval_transitions=10_000,
):
    # actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2 ** 32)))
    # trajs = []
    # stats = defaultdict(list)
    batch = dataset.sample(num_eval_transitions)

    rng = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))
    rng, seed = jax.random.split(rng)
    stats = estimator.evaluate_estimation(batch, seed=seed)

    # renders = []
    # for i in trange(num_eval_episodes + num_video_episodes):
    #     traj = defaultdict(list)
    #     should_render = i >= num_eval_episodes
    #
    #     observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render))
    #     goal = info.get('goal')
    #     goal_frame = info.get('goal_rendered')
    #     done = False
    #     step = 0
    #     render = []
    #     while not done:
    #         action = actor_fn(observations=observation, goals=goal, temperature=eval_temperature)
    #         action = np.array(action)
    #         # if not config.get('discrete'):
    #         #     action = np.clip(action, -1, 1)
    #
    #         next_observation, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         step += 1
    #
    #         if should_render and (step % video_frame_skip == 0 or done):
    #             frame = env.render().copy()
    #             if goal_frame is not None:
    #                 render.append(np.concatenate([goal_frame, frame], axis=0))
    #             else:
    #                 render.append(frame)
    #
    #         transition = dict(
    #             observation=observation,
    #             next_observation=next_observation,
    #             action=action,
    #             reward=reward,
    #             done=done,
    #             info=info,
    #         )
    #         add_to(traj, transition)
    #         observation = next_observation
    #     if i < num_eval_episodes:
    #         add_to(stats, flatten(info))
    #         trajs.append(traj)
    #     else:
    #         renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.asarray(v)

    return stats


def evaluate_heatmaps(
        estimator,
        dataset,
        env,
        num_heatmaps,
        num_grid_x=50,
        num_grid_y=50,
):
    unwrapped_env = env.unwrapped
    tree = ET.parse(unwrapped_env.fullpath)
    worldbody = tree.find('.//worldbody')

    wall_positions = []
    wall_sizes = []
    for element in worldbody.findall(".//geom[@material='wall']"):
        pos = [float(s) for s in element.get('pos').split(' ')][:2]
        size = [float(s) for s in element.get('size').split(' ')][:2]

        wall_positions.append(pos)
        wall_sizes.append(size)

    wall_positions = np.array(wall_positions)
    wall_sizes = np.array(wall_sizes)
    world_ranges = np.array([wall_positions.min(axis=0), wall_positions.max(axis=0)]).T

    env_info = {
        'wall_positions': wall_positions,
        'wall_sizes': wall_sizes,
        'world_ranges': world_ranges,
    }

    grid_x = np.linspace(*world_ranges[0], num_grid_x)
    grid_y = np.linspace(*world_ranges[1], num_grid_y)

    mesh_x, mesh_y = np.array(np.meshgrid(grid_x, grid_y))
    mesh_grid_xys = np.stack([mesh_x, mesh_y], axis=-1)

    batch = dataset.sample(num_heatmaps)
    rng = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))
    rng, seed = jax.random.split(rng)

    observations = batch['observations'][:, None].repeat(
        num_grid_x * num_grid_y, axis=1).reshape(
        [-1, env.observation_space.shape[-1]])
    goals = mesh_grid_xys[None].repeat(num_heatmaps, axis=0).reshape(
        [-1, env.observation_space.shape[-1]])
    values = estimator.compute_values(observations, goals, seed=seed)
    values = values.reshape(num_heatmaps, num_grid_x, num_grid_y)

    value_info = {
        'observations': batch['observations'],
        'mesh_grid_xys': mesh_grid_xys,
        'mesh_x': mesh_x,
        'mesh_y': mesh_y,
        'values': values
    }

    return value_info, env_info
