import os
import os.path as osp

import h5py
from absl import app, flags

from dataset_aggregator import make_dataset


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'env_name', 'Environment name.')
flags.DEFINE_string('expl_agent_name', 'rnd', 'Exploratory agent name.')
flags.DEFINE_string('dataset_dir', '~/.exorl/expl_datasets', 'Download the dataset to this directory.')
flags.DEFINE_string('save_path', None, 'Save the dataset to this path.')
flags.DEFINE_integer('num_workers', 8, 'Number of workers to collect transitions.')
flags.DEFINE_integer('skip_size', 0, 'Number of transitions to skip in the dataset.')
flags.DEFINE_integer('dataset_size', 5_000_000, 'Size of the dataset.')
flags.DEFINE_integer('relabel_reward', 0, 'Whether to relabel the reward of each transition.')


def main(_):
    # create data storage
    if FLAGS.env_name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
    else:
        domain = FLAGS.env_name.split('_', 1)[0]
    dataset_dir = osp.expanduser(FLAGS.dataset_dir)
    dataset_dir = osp.join(dataset_dir, domain, FLAGS.expl_agent_name, 'buffer')
    print(f'dataset dir: {dataset_dir}')

    dataset = make_dataset(
        FLAGS.env_name, dataset_dir,
        FLAGS.skip_size, FLAGS.dataset_size,
        FLAGS.num_workers,
        FLAGS.relabel_reward
    )

    save_path = osp.expanduser(FLAGS.save_path)
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with h5py.File(save_path, 'w') as f:
        for k, v in dataset.items():
            f.create_dataset(k, data=v)

    print("Save dataset to: {}".format(save_path))


if __name__ == '__main__':
    app.run(main)
