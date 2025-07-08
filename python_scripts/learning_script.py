#!/usr/bin/env python3

import os
from datetime import datetime
import argparse
from stable_baselines3 import PPO, SAC, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes, StopTrainingOnRewardThreshold, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')


def results_directory(base_directory, results_id):
    path = os.path.join(base_directory, 'save-' + results_id + '-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))

    if not os.path.exists(path):
        os.makedirs(path + '/')

    return str(path)


def get_model(model_class, environment, path, reuse_model=False, seed: int = 0, **kwargs):
    if reuse_model:
        return model_class.load(
            path=path,
            device='auto',
            env=environment,
            force_reset=True
        )

    return model_class(
        'MlpPolicy',
        environment,
        tensorboard_log=path + '/tb/',
        seed=seed,
        verbose=0,
        device='auto',
        **kwargs
    )


def callbacks(evaluation_environment, parallel_environments, path_to_results, stop_on_max_episodes:dict, stop_on_reward_threshold:dict, save_checkpoints:dict):
    eval_callback = EvalCallback(evaluation_environment,
                                 verbose=0,
                                 best_model_save_path=path_to_results + '/',
                                 log_path=path_to_results + '/',
                                 eval_freq=int(10000 / parallel_environments),
                                 deterministic=True,
                                 render=False)

    callback_list = []

    if stop_on_reward_threshold['stop']:
        stop_on_reward_threshold_callback = StopTrainingOnRewardThreshold(stop_on_reward_threshold['threshold'], verbose=0)
        eval_callback = EvalCallback(evaluation_environment,
                                     callback_on_new_best=stop_on_reward_threshold_callback,
                                 verbose=0,
                                 best_model_save_path=path_to_results + '/',
                                 log_path=path_to_results + '/',
                                 eval_freq=int(10000 / parallel_environments),
                                 deterministic=True,
                                 render=False)
    elif stop_on_max_episodes['stop']:
        stop_on_max_episodes_callback = StopTrainingOnMaxEpisodes(int(stop_on_max_episodes['episodes'] / parallel_environments), verbose=1)
        callback_list.append(stop_on_max_episodes_callback)

    if save_checkpoints['save']:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_checkpoints['save_frequency'],
            save_path=path_to_results + '/checkpoints/',
            name_prefix='checkpoint',
            save_replay_buffer=True,
            save_vecnormalize=True
        )
        callback_list.append(checkpoint_callback)

    callback_list.append(eval_callback)

    return callback_list


def run_learning(environment,
                 learning_id,
                 algorithm='ppo',
                 continuous_learning=False,
                 path_to_previous_model=None,
                 parallel_environments=4,
                 time_steps=10e7,
                 seed=0,
                 stop_on_max_episodes=None,
                 stop_on_reward_threshold=None,
                 save_checkpoints=None,
                 output_directory=DEFAULT_OUTPUT_FOLDER
                 ):

    path_to_results = results_directory(output_directory, learning_id)

    learning_environment = make_vec_env(environment,
                                        n_envs=parallel_environments,
                                        vec_env_cls=SubprocVecEnv
                                        )
    evaluation_environment = make_vec_env(environment,
                                          n_envs=parallel_environments,
                                          vec_env_cls=SubprocVecEnv
                                          )

    model_map = {
        'ppo': (
            PPO, {
                'batch_size': 256,
                'learning_rate': 2e-4,
                'n_steps': 8192,
                'n_epochs': 6,
                'clip_range': 0.12,
                'ent_coef': 0.001
            }
        ),
        'sac': (SAC, {}),
        'ddpg': (DDPG, {}),
        'td3': (
            TD3, {
                'batch_size': 256,
                'learning_rate': 10e-3,
            }
        )
    }

    try:
        model_class, extra_args = model_map[algorithm]
    except KeyError:
        raise ValueError(f"{algorithm} is not supported.")

    model = get_model(
        model_class,
        learning_environment,
        f'{path_to_previous_model}/best_model' if continuous_learning else path_to_results,
        reuse_model=continuous_learning,
        seed=seed,
        **extra_args
    )

    callback_list = callbacks(evaluation_environment, parallel_environments, path_to_results,
                              stop_on_max_episodes, stop_on_reward_threshold, save_checkpoints)

    model.learn(total_timesteps=int(time_steps),
                callback=callback_list,
                log_interval=1,
                progress_bar=False)

    model.save(path_to_results + '/final_model.zip')

    return path_to_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent RL learning")
    parser.add_argument(
        '--env_name',
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--algorithm',
        default='ppo',
        choices=['ppo', 'sac', 'ddpg'],
        type=str,
        help='The algorithm for training (ppo, sac, ddpg)'
    )
    parser.add_argument(
        '--output_directory',
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument(
        '--env_parameters',
        default=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
        help="Parameters for the environment to learn"
    )
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random seed for reproducibility'
    )

    results_path = run_learning(**vars(parser.parse_args()))
    print(f" #### The training process has end, the best policy was saved in: {results_path} ####")
