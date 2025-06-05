import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import matplotlib.pyplot as plt
import numpy as np
import wandb
import wandb_osh
from wandb_osh.hooks import TriggerWandbSyncHook
from PIL import Image, ImageEnhance


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    wandb_output_dir=tempfile.mkdtemp(),
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    # wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    trigger_sync = None
    if mode == 'offline':
        wandb_osh.set_log_level("ERROR")
        trigger_sync = TriggerWandbSyncHook()

    return run, trigger_sync


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')


def get_wandb_heatmaps(value_info, env_info, num_max_cols=8):
    # get walls from the env
    # unwrapped_env = env.unwrapped
    # tree = ET.parse(unwrapped_env.fullpath)
    # worldbody = tree.find('.//worldbody')
    #
    # wall_positions = []
    # wall_sizes = []
    # for element in worldbody.findall(".//geom[@material='wall']"):
    #     pos = [float(s) for s in element.get('pos').split(' ')][:2]
    #     size = [float(s) for s in element.get('size').split(' ')][:2]
    #
    #     wall_positions.append(pos)
    #     wall_sizes.append(size)
    #
    # wall_positions = np.array(wall_positions)
    # wall_sizes = np.array(wall_sizes)
    # world_ranges = np.array([wall_positions.min(axis=0), wall_positions.max(axis=0)]).T

    # create the meshgrid
    # num_grid_x = 50
    # num_grid_y = 50
    # grid_x = np.linspace(*world_ranges[0], num_grid_x)
    # grid_y = np.linspace(*world_ranges[1], num_grid_y)
    #
    # mesh_x, mesh_y = np.array(np.meshgrid(grid_x, grid_y))
    # mesh_grid_xys = np.stack([mesh_x, mesh_y], axis=-1)

    # sample trajectories from the dataset
    # initial_state_idxs = dataset.initial_locs[
    #     np.searchsorted(dataset.initial_locs, np.arange(dataset.size), side='right') - 1]
    # final_state_idxs = dataset.terminal_locs[
    #     np.searchsorted(dataset.terminal_locs, np.arange(dataset.size))]
    #
    # traj_start_idxs = np.unique(initial_state_idxs)
    # traj_end_idxs = np.unique(final_state_idxs)
    # num_trajs = len(traj_start_idxs)
    #
    # if traj_idxs is None:
    #     traj_idxs = np.random.randint(num_trajs, size=n_heatmaps)
    #
    # traj_observations = []
    # for (traj_start_idx, traj_end_idx) in zip(traj_start_idxs[traj_idxs], traj_end_idxs[traj_idxs]):
    #     observations = dataset['observations'][traj_start_idxs:traj_end_idx + 1]
    #
    #     traj_observations.append(observations)
    # traj_observations = np.array(traj_observations)

    # rng = jax.random.PRNGKey(np.random.randint(0, 2 ** 32))
    # rng, seed = jax.random.split(rng)
    # stats = estimator.evaluate_estimation(batch, seed=seed)
    # batch = dataset.sample(n_heatmaps)

    wall_positions, wall_sizes, world_ranges = (
        env_info['wall_positions'], env_info['wall_sizes'], env_info['world_ranges'])
    # import matplotlib.patches as patches

    # rects = []
    # for wall_pos, wall_size in zip(wall_positions, wall_sizes):
    #     rect = plt.Rectangle(wall_pos - wall_size, wall_size[0] * 2, wall_size[1] * 2)
    #     rects.append(rect)

    mesh_x, mesh_y, observations, values = (
        value_info['mesh_x'], value_info['mesh_y'], value_info['observations'], value_info['values'])
    num_heatmaps = len(values)
    num_rows = num_heatmaps // num_max_cols
    if num_rows < 1:
        num_cols = num_heatmaps % num_max_cols
    else:
        num_cols = num_max_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_max_cols,
                             figsize=(4.2 * num_cols, 4.4 * num_rows))
    for observation, value, ax in zip(observations, values, axes.flatten()):

        contour = ax.contourf(mesh_x, mesh_y, np.exp(value))
        plt.colorbar(contour, ax=ax)

        for wall_pos, wall_size in zip(wall_positions, wall_sizes):
            rect = plt.Rectangle(wall_pos - wall_size, wall_size[0] * 2, wall_size[1] * 2,
                                 color='white')
            ax.add_patch(rect)

        ax.scatter(observation[0], observation[1], s=20.0, marker='x', color='red', label=r'$s_0$')

        ax.set(title='value',
               xlabel='x', ylabel='y',
               xlim=[*world_ranges[0]], ylim=[*world_ranges[1]])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)

    fig.tight_layout()

    return wandb.Image(fig)
