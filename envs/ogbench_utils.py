import os

import gymnasium
import numpy as np

from ogbench.utils import DEFAULT_DATASET_DIR, download_datasets, load_dataset
from ogbench.relabel_utils import add_oracle_reps, relabel_dataset


def make_env_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    compact_dataset=False,
    env_only=False,
    add_info=False,
    **env_kwargs,
):
    """Make OGBench environment and load datasets.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Directory to save the datasets.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').
        env_only: Whether to return only the environment.
        add_info: Whether to add observation information ('qpos', 'qvel', and 'button_states') to the datasets.
        **env_kwargs: Keyword arguments to pass to the environment.
    """
    # Make environment.
    splits = dataset_name.split('-')
    dataset_add_info = add_info
    if 'singletask' in splits:
        # Single-task environment.
        pos = splits.index('singletask')
        if 'ft' in splits:
            env_name = '-'.join(splits[: pos - 2] + splits[pos:])  # Remove the dataset type.
        else:
            env_name = '-'.join(splits[: pos - 1] + splits[pos:])  # Remove the dataset type.
        env = gymnasium.make(env_name, **env_kwargs)
        dataset_name = '-'.join(splits[:pos] + splits[-1:])  # Remove the words 'singletask' and 'task\d' (if exists).
        dataset_add_info = True
    elif 'oraclerep' in splits:
        # Environment with oracle goal representations.
        env_name = '-'.join(splits[:-3] + splits[-1:])  # Remove the dataset type and the word 'oraclerep'.
        env = gymnasium.make(env_name, use_oracle_rep=True, **env_kwargs)
        dataset_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the word 'oraclerep'.
        dataset_add_info = True
    else:
        # Original, goal-conditioned environment.
        env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
        env = gymnasium.make(env_name, **env_kwargs)

    if env_only:
        return env

    # Load datasets.
    dataset_dir = os.path.expanduser(dataset_dir)
    download_datasets([dataset_name], dataset_dir)
    train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
    val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    ob_dtype = np.uint8 if ('visual' in env_name or 'powderworld' in env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in env_name else np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
        add_info=dataset_add_info,
    )

    if 'singletask' in splits:
        # Add reward information to the datasets.
        relabel_dataset(env_name, env, train_dataset)
        relabel_dataset(env_name, env, val_dataset)

    if 'oraclerep' in splits:
        # Add oracle goal representations to the datasets.
        add_oracle_reps(env_name, env, train_dataset)
        add_oracle_reps(env_name, env, val_dataset)

    if not add_info:
        # Remove information keys.
        for k in ['qpos', 'qvel', 'button_states']:
            if k in train_dataset:
                del train_dataset[k]
            if k in val_dataset:
                del val_dataset[k]

    return env, train_dataset, val_dataset
