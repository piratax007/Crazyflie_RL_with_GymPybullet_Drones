#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
from environments.CurriculumStage1 import CurriculumStage1
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool, FIRFilter


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(policy_path, model):
    # if os.path.isfile(policy_path + '/best_model.zip'):
    if os.path.isfile(policy_path + '/' + model):
        # return PPO.load(policy_path + '/best_model.zip')
        return PPO.load(policy_path + '/' + model)

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def spiral_trajectory(number_of_points: int = 50, radius: int = 2, angle_range: float = 30.0) -> tuple:
    angles = np.linspace(0, 4 * np.pi, number_of_points)
    x_coordinates = radius * np.cos(angles)
    y_coordinates = radius * np.sin(angles)
    z_coordinates = np.linspace(0, 1, number_of_points)

    yaw_angles = np.arctan2(y_coordinates, x_coordinates)

    angle_range_rad = np.radians(angle_range)

    oscillation = angle_range_rad * np.sin(np.linspace(0, 4 * np.pi, number_of_points))

    yaw_angles += oscillation

    yaw_angles = np.clip(yaw_angles, -angle_range_rad, angle_range_rad)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles


def smooth_trajectory(points, num_points=100):
    points = np.array(points)
    tck, u = splprep([points[:,0], points[:,1], points[:,2]], s=0)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    smooth_points = np.vstack((x_fine, y_fine, z_fine)).T

    tangents = np.diff(smooth_points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    tangents = np.vstack((tangents, tangents[-1]))

    roll_pitch_yaw = []

    for i in range(len(smooth_points)):
        t = tangents[i]
        y_axis = t
        z_axis = np.array([0, 0, 1])
        if np.allclose(y_axis, z_axis):
            z_axis = np.array([0, 1, 0])
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)

        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        roll_pitch_yaw.append((roll, pitch, yaw))

    x_tuple = tuple(smooth_points[:, 0])
    y_tuple = tuple(smooth_points[:, 1])
    z_tuple = tuple(smooth_points[:, 2])
    yaw_tuple = tuple([yaw for _, _, yaw in roll_pitch_yaw])

    return x_tuple, y_tuple, z_tuple, yaw_tuple


def run_simulation(
        test_env,
        policy_path,
        model='best_model.zip',
        gui=True,
        record_video=True,
        reset=False,
        save=False,
        plot=False,
        debug=False,
        apply_filter=False,
        comment=""
):
    policy = get_policy(policy_path, model)

    test_env = test_env(gui=gui,
                        obs=ObservationType('kin'),
                        act=ActionType('rpm'),
                        # initial_xyzs=np.array([[0, 0, 0]]),
                        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset(seed=42, options={})
    simulation_length = (test_env.EPISODE_LEN_SEC + 5) * test_env.CTRL_FREQ

    start = time.time()

    firfilter = FIRFilter()
    #
    for _ in range(firfilter.buffer_size):
        firfilter.buffer.append(np.zeros((1, 4)))

    # x_straight = np.linspace(-2, 2, simulation_length)
    # y_straight = np.linspace(-2, 2, simulation_length)
    x_target, y_target, z_target, yaw_target = spiral_trajectory(simulation_length, 2)
    # points = [
    #     [-4, 0, 0],
    #     [-2, 1, 0.75],
    #     [-1, -2, 1.5],
    #     [1, 0, 2],
    #     [4, 1, 1],
    #     [6, -1, 0.5]
    # ]
    # x_target, y_target, z_target, yaw_target = smooth_trajectory(points, num_points=simulation_length)

    for i in range(simulation_length):
        # WAY-POINT TRACKING
        # if i < simulation_length / 5:
        #     x_target = -1
        #     y_target = 1
        #     z_target = 0
        #     yaw_target = -0.52
        # elif simulation_length / 5 < i < 2 * simulation_length / 5:
        #     x_target = -2
        #     y_target = 0
        #     z_target = 0.5
        #     yaw_target = 0
        # elif 2 * simulation_length / 5 < i < 3 * simulation_length / 5:
        #     x_target = -2
        #     y_target = -2
        #     z_target = 1.5
        #     yaw_target = 0.35
        # elif 3 * simulation_length / 5 < i < 4 * simulation_length / 5:
        #     x_target = -1
        #     y_target = -3
        #     z_target = 0
        #     yaw_target = 0.7
        # else:
        #     x_target = -2.8
        #     y_target = -3.8
        #     z_target = 1
        #     yaw_target = 0
        #
        # obs[0][0] -= x_target
        # obs[0][1] -= y_target
        # obs[0][2] -= z_target
        # obs[0][5] -= 1

        # TRAJECTORY TRACKING
        # obs[0][0] -= x_target[i]
        # obs[0][1] -= y_target[i]
        # obs[0][2] -= z_target[i]
        # obs[0][5] -= yaw_target[i]

        action, _states = policy.predict(obs,
                                         deterministic=True
                                         )

        if apply_filter:
            action = firfilter.filter_actions(action)

        obs, reward, terminated, truncated, info = test_env.step(action)
        actions = test_env._getDroneStateVector(0)[16:20]
        actions2 = actions.squeeze()
        obs2 = obs.squeeze()

        if debug:
            print(f"""
            #################################################################
            Observation Space:
            Position: {obs[0][0:3]}
            Orientation: {in_degrees(obs[0][3:6])}
            Linear Velocity: {obs[0][6:9]}
            Angular Velocity: {obs[0][9:12]}
            -----------------------------------------------------------------
            Action Space: type {type(action)} value {action}
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
                             np.zeros(4),
                             obs2[3:12],
                             actions2
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
        # logger.plot_rpms()
        # logger.plot_trajectory()

    if save:
        logger.save_as_csv(comment)
        # save_to_csv(tuple([log_timestamp, log_reward]), policy_path, 'instantaneous_reward', 'full-task')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a simulation given a trained policy")
    parser.add_argument(
        '--policy_path',
        help='The path to a zip file containing the trained policy'
    )
    parser.add_argument(
        '--model',
        help='The zip file containing the trained policy'
    )
    parser.add_argument(
        '--test_env',
        default=CurriculumStage1,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--gui',
        default=True,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--record_video',
        default=False,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--reset',
        default=False,
        type=str2bool,
        help="If you want to reset the environment, every time that the drone achieve the target position"
    )
    parser.add_argument(
        '--save',
        default=False,
        type=str2bool,
        help='Allow to save the trained data using csv and npy files'
    )
    parser.add_argument(
        '--comment',
        default="",
        type=str,
        help="A comment to describe de simulation saved data"
    )
    parser.add_argument(
        '--plot',
        default=False,
        type=str2bool,
        help="If are shown demo plots"
    )
    parser.add_argument(
        '--debug',
        default=False,
        type=str2bool,
        help="Prints debug information"
    )
    parser.add_argument(
        '--apply_filter',
        default=False,
        type=str2bool,
        help="Applies a low pass to the actions coming from the policy"
    )

    run_simulation(**vars(parser.parse_args()))
