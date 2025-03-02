#!/usr/bin/env python3
import argparse
from environments.ejc_cl_stage1 import EjcCLStage1
from environments.ejc_cl_stage2 import EjcCLStage2
from environments.ejc_cl_stage3 import EjcCLStage3
from environments.basic_reward import BasicReward
from environments.basic_reward_second_stage import BasicRewardSecondStage
from environments.exploration_reward_contribution import ExplorationRewardContribution
from environments.stability_reward_contribution import StabilityRewardContribution
from environments.exploration_reward_out import ExplorationRewardOut
from environments.stability_reward_out import StabilityRewardOut
from environments.navigation_reward_out import NavigationRewardOut
from environments.ejc_cl_stage1_scaled_reward import EjcCLStage1ScaledReward
from environments.ejc_cl_stage2_scaled_reward import EjcCLStage2ScaledReward
from environments.ejc_cl_stage3_scaled_reward import EjcCLStage3ScaledReward
from python_scripts.learning_script import run_learning
from gym_pybullet_drones.utils.utils import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Agent learning')
    parser.add_argument(
        '--environment',
        default='EjcCLStage1',
        type=str,
        choices=[
            'EjcCLStage1',
            'EjcCLStage2',
            'EjcCLStage3',
            'BasicReward',
            'ExplorationRewardOut',
            'StabilityRewardOut',
            'NavigationRewardOut',
            'BasicRewardSecondStage',
            'ExplorationRewardContribution',
            'StabilityRewardContribution',
            'EjcCLStage1ScaledReward',
            'EjcCLStage2ScaledReward',
            'EjcCLStage3ScaledReward'
        ],
        help='An imported environment'
    )
    parser.add_argument(
        '--learning-id',
        type=str,
        help='A string to be added to the results directory name'
    )
    parser.add_argument(
        '--algorithm',
        default='ppo',
        type=str,
        help='The algorithm for training (ppo, sac, ddpg)'
    )
    parser.add_argument(
        '--continuous-learning',
        default=False,
        type=str2bool,
        help='A boolean indicating whether to use continuous learning or not'
    )
    parser.add_argument(
        '--path-to-previous-model',
        default=None,
        type=str,
        help='A string indicating the path to the previous model'
    )
    parser.add_argument(
        '--parallel-environments',
        default=4,
        type=int,
        help='The number of parallel environments (1 for ddpg, 4 suggested for ppo and sac)'
    )
    parser.add_argument(
        '--time-steps',
        default=int(10e7),
        type=int,
        help='The number of time steps'
    )
    parser.add_argument(
        '--seed',
        default=90,
        type=int,
        help='Seed for reproducibility'
    )
    parser.add_argument(
        '--stop-on-max-episodes-flag',
        default=False,
        type=str2bool,
        help='A boolean indicating whether to stop on max episodes'
    )
    parser.add_argument(
        '--stop-episodes',
        default=30000000,
        type=int,
        help='The number of episodes to stop on'
    )
    parser.add_argument(
        '--stop-on-reward-threshold-flag',
        default=False,
        type=str2bool,
        help='A boolean indicating whether to stop on reward threshold'
    )
    parser.add_argument(
        '--stop-on-reward-threshold-threshold',
        default=6000.,
        type=float,
        help='A float indicating whether to stop on reward threshold'
    )

    args = parser.parse_args()

    environment_map = {
        'EjcCLStage1': EjcCLStage1,
        'EjcCLStage2': EjcCLStage2,
        'EjcCLStage3': EjcCLStage3,
        'BasicReward': BasicReward,
        'ExplorationRewardOut': ExplorationRewardOut,
        'StabilityRewardOut': StabilityRewardOut,
        'NavigationRewardOut': NavigationRewardOut,
        'BasicRewardSecondStage': BasicRewardSecondStage,
        'ExplorationRewardContribution': ExplorationRewardContribution,
        'StabilityRewardContribution': StabilityRewardContribution,
        'EjcCLStage1ScaledReward': EjcCLStage1ScaledReward,
        'EjcCLStage2ScaledReward': EjcCLStage2ScaledReward,
        'EjcCLStage3ScaledReward': EjcCLStage3ScaledReward
    }

    environment_class = environment_map.get(args.environment)
    if environment_class is None:
        raise ValueError(f'Invalid environment: {args.environment}')

    stop_episodes_flag = args.stop_on_max_episodes_flag
    stop_episodes = args.stop_episodes
    reward_stop_flag = args.stop_on_reward_threshold_flag
    reward_stop_threshold = args.stop_on_reward_threshold_threshold

    print(f"""
    ################# Learning Started ########################
    Learning ID: {args.learning_id}
    Seed: {args.seed}
    #######################################################
    """)

    results = run_learning(environment=environment_class,
                           learning_id=args.learning_id,
                           algorithm=args.algorithm,
                           continuous_learning=args.continuous_learning,
                           path_to_previous_model=args.path_to_previous_model,
                           parallel_environments=args.parallel_environments,
                           time_steps=int(args.time_steps),
                           seed=args.seed,
                           stop_on_max_episodes=dict(stop=stop_episodes_flag, episodes=stop_episodes),
                           stop_on_reward_threshold=dict(stop=reward_stop_flag, threshold=reward_stop_threshold),
                           save_checkpoints=dict(save=True, save_frequency=250000)
                           )

    print(f"""
    ################# Learning End ########################
    Results: {results}
    #######################################################
    """)
