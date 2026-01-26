#!/usr/bin/env python3
import argparse
import time
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3
from python_scripts.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool
from environments import environment_map
from python_scripts.simulation_helpers import get_policy, in_degrees

def run_simulation(
        test_env,
        policy_path,
        algorithm='ppo',
        model='best_model.zip',
        gui=True,
        record_video=False,
        simulation_length=20,
        reset=False,
        save=False,
        plot=False,
        debug=False,
        comment=""
):
    """
    Runs a simulation using the provided environment, policy, and specified parameters.

    The function initializes the test environment with specific configurations, prepares the policy using
    the selected reinforcement learning algorithm and model, and iteratively steps through the simulation while
    logging data, rendering the GUI, and handling state updates. It supports recording videos of the simulation,
    debugging outputs, and optional resetting when the simulation terminates. Additionally, it provides options
    to plot and save logged data after the simulation ends.

    Args:
        test_env: The environment class to be used for simulation.
        policy_path: Path to the directory containing the policy model.
        algorithm: The RL algorithm to use for simulation (default: 'ppo').
        model: The specific model file to load within the policy directory (default: 'best_model.zip').
        gui: Whether to enable GUI rendering during simulation (default: True).
        record_video: Whether to record the video of the simulation (default: False).
        simulation_length: Length of the simulation in seconds (default: 20).
        reset: Whether to reset the environment upon termination (default: False).
        save: Whether to save logged data as a CSV (default: False).
        plot: Whether to plot logged data graphs after simulation (default: False).
        debug: Whether to enable detailed debug outputs during simulation (default: False).
        comment: Additional comments to include in the saved logs (default: "").
        obs_delay_s: Seconds to delay the observation vector by (default: 0.0).
        wind: Dictionary containing wind parameters for simulation (default: None).

    Raises:
        KeyError: If the specified algorithm is not available in the model_map dictionary.

    Returns:
        None

    How To:
        If you want the change the initial position of the camera, define it before the simulation loop:
        p.resetDebugVisualizerCamera(1, 125, -10, [1, 1, 1])
    """

    model_map = {
        'ppo': PPO,
        'sac': SAC,
        'ddpg': DDPG,
        'td3': TD3
    }

    policy = get_policy(model_map[algorithm], policy_path, model)

    test_env = test_env(
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        gui=gui,
        observation_space=ObservationType('kin'),
        action_space=ActionType('rpm'),
        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset()

    simulation_seconds = simulation_length * test_env.CTRL_FREQ

    start = time.time()

    for i in range(simulation_seconds):
        obs[0][0] += 0.04
        obs[0][1] -= 0.04
        clipped_actions, _states = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(clipped_actions)

        clipped_rpm = test_env._getDroneStateVector(0)[16:20].squeeze()
        quaternion = test_env._getDroneStateVector(0)[3:7]
        obs2 = obs.squeeze()

        if debug:
            print(f"""
            #################################################################
            Observations:
            Position: {obs[0][0:3]}
            Orientation: {in_degrees(obs[0][3:6])}
            Linear Velocity: {obs[0][6:9]}
            Angular Velocity: {obs[0][9:12]}
            -----------------------------------------------------------------
            Raw Actions Clipped: type {type(clipped_actions)} value {clipped_actions}
            Raw RPM Clipped: {clipped_rpm}
            Terminated: {terminated}
            Truncated: {truncated}
            -----------------------------------------------------------------
            Policy Architecture: {policy.policy}
            #################################################################
            """)

        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3],
                             quaternion,
                             obs2[3:12],
                             clipped_rpm
                             ]),
            reward=reward,
            control=np.zeros(12)
        )

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if reset and terminated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

    if plot:
        logger.plot_position_and_orientation()
        # logger.plot_instantaneous_reward()
        # logger.plot()
        # logger.plot_pwms()
        # logger.plot_trajectory()

    if save:
        logger.save_as_csv(comment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a simulation given a trained policy")
    parser.add_argument('--policy_path', help='The path to a zip file containing the trained policy')
    parser.add_argument('--model', help='The zip file containing the trained policy')
    parser.add_argument('--algorithm', default='ppo', help='The algorithm used for training')
    parser.add_argument('--test_env', default='CLStage1Sim2Real', type=str,help='The name of the environment to learn, registered with gym_pybullet_drones')
    parser.add_argument('--simulation-length', default=20, type=int, help='The length of the simulation in seconds')
    parser.add_argument('--reset', default=False, type=str2bool, help="If you want to reset the environment, every time that the drone achieve the target position")
    parser.add_argument('--save', default=False, type=str2bool, help='Allow to save the trained data using csv and npy files')
    parser.add_argument('--comment', default="", type=str, help="A comment to describe de simulation saved data")
    parser.add_argument('--plot', default=False, type=str2bool, help="If are shown demo plots")
    parser.add_argument('--debug', default=False, type=str2bool, help="Prints debug information")
    parser.add_argument('--record-video', default=False, type=str2bool, help="Record simulation video")

    args = parser.parse_args()

    environment_class = environment_map.get(args.test_env)
    if environment_class is None:
        raise ValueError(f"Unknown environment: {args.test_env}")

    run_simulation(
        test_env=environment_class,
        policy_path=args.policy_path,
        algorithm=args.algorithm,
        model=args.model,
        gui=True,
        record_video=args.record_video,
        simulation_length=args.simulation_length,
        reset=args.reset,
        save=args.save,
        comment=args.comment,
        plot=args.plot,
        debug=args.debug
    )
