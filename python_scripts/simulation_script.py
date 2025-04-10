#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import pybullet as p
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO, SAC, DDPG, TD3
from environments.ObS12Stage1 import ObS12Stage1
from environments.ObS12Stage2 import ObS12Stage2
from environments.ObS12Stage3 import ObS12Stage3
from environments.basic_reward import BasicReward
from environments.ejc_cl_stage1 import EjcCLStage1
from environments.ejc_cl_stage2 import EjcCLStage2
from environments.ejc_cl_stage3 import EjcCLStage3
from environments.ejc_cl_stage1_scaled_reward import EjcCLStage1ScaledReward
from environments.ejc_cl_stage2_scaled_reward import EjcCLStage2ScaledReward
from environments.ejc_cl_stage3_scaled_reward import EjcCLStage3ScaledReward
from environments.exploration_reward_out_30hz import ExplorationRewardOut30Hz
from environments.exploration_reward_out import ExplorationRewardOut
from environments.stability_reward_out_30hz import StabilityRewardOut30Hz
from environments.stability_reward_out_30hz_stage_2 import StabilityRewardOut30HzStage2
from environments.stability_reward_out import StabilityRewardOut
from environments.stability_reward_out_stage2_39 import StabilityRewardOutStage2
from environments.target_reward_out_30hz_39 import TargetRewardOut30Hz
from environments.target_reward_out_200hz_39 import TargetRewardOut200Hz
from environments.WithoutCurriculumLearning_200Hz import WithoutCurriculumLearning200Hz
from environments.CL_Stage1_Sim2Real import CLStage1Sim2Real
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(model_class, policy_path, model):
    if os.path.isfile(policy_path + '/' + model):
        return model_class.load(policy_path + '/' + model)

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def helix_trajectory(number_of_points: int = 50, radius: int = 2, angle_range: float = 30.0) -> tuple:
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


def lemniscata_trajectory(number_of_points: int = 50, a: float = 2) -> tuple:
    t = np.linspace(0, 2.125 * np.pi, number_of_points)
    x_coordinates = a * np.sin(t) / (1 + np.cos(t) ** 2)
    y_coordinates = a * np.sin(t) * np.cos(t) / (1 + np.cos(t) ** 2)
    z_coordinates = np.zeros(number_of_points)

    yaw_angles = np.arctan2(-y_coordinates, -x_coordinates)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles


def smooth_trajectory(points, num_points=100):
    points = np.array(points)
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0)
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


def random_cylindrical_positions(
        inner_radius: float = 0.0,
        outer_radius: float = 1.5,
        cylinder_height: float = 1.5,
        cylinder_center: tuple = (0, 0, 1),
        mode: str = "inside",
        min_distance: float = 0.0,
        max_distance: float = 0.0
) -> tuple:
    cx, cy, cz = cylinder_center

    if mode == "inside":
        r = np.sqrt(np.random.uniform(inner_radius ** 2, outer_radius ** 2))
    elif mode == "outside":
        r = np.sqrt(np.random.uniform((outer_radius + min_distance) ** 2, (outer_radius + max_distance) ** 2))
    else:
        r = 0

    theta = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(-cylinder_height / 2, cylinder_height / 2 + max_distance)

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = cz + z

    return x, y, z


def add_way_point(position, radius=0.1, color=(1, 0, 0, 0.5)):
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
    )
    sphere_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=sphere_visual,
        basePosition=position,
    )
    return sphere_id


def run_simulation(
        test_env,
        policy_path,
        algorithm='ppo',
        model='best_model.zip',
        gui=True,
        record_video=True,
        reset=False,
        save=False,
        plot=False,
        debug=False,
        comment=""
):

    model_map = {
        'ppo': PPO,
        'sac': SAC,
        'ddpg': DDPG,
        'td3': TD3
    }

    policy = get_policy(model_map[algorithm], policy_path, model)

    test_env = test_env(
        initial_xyzs=np.array([[0.0, 0.0, 0.0]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        # initial_xyzs=np.array([[-1, 1, 0]]),
        # initial_rpys=np.array([[0.0, 0.0, 0.78]]),
        # target_rpys=np.array([0.0, 0.0, 0.0]),
        # initial_xyzs = np.array([[*random_cylindrical_positions(outer_radius=2.0, cylinder_height=2, mode='inside')]]),
        # initial_rpys = np.array([[
        #         np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
        #         np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
        #         np.random.uniform(-3.14, 3.14 + 1e-10, 1)[0]
        #     ]]),
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
    simulation_length = (test_env.EPISODE_LENGTH_SECONDS + 15) * test_env.CTRL_FREQ

    start = time.time()

    # p.resetDebugVisualizerCamera(1, 125, -10, [1, 1, 1])
    # p.resetDebugVisualizerCamera(1, 0, -90, [-2, -1.5, 3])
    # _ = {
    #     0: add_way_point((-1, 1, 0), radius=0.025),
    #     1: add_way_point((-1, 1, 1), radius=0.025),
    #     2: add_way_point((-2, 0, 1.5), radius=0.025),
    #     3: add_way_point((-2, -2, 2.5), radius=0.025),
    #     4: add_way_point((-1, -3, 1), radius=0.025),
    #     5: add_way_point((-3, -3.5, 2), radius=0.025),
    # }

    # x_target, y_target, z_target, yaw_target = spiral_trajectory(simulation_length, 2)
    # x_target, y_target, z_target, yaw_target = spiral_trajectory(simulation_length, 2)

    for i in range(simulation_length):
        # obs[0][0] += 0.025
        # obs[0][1] -= 0.023
        # obs[0][2] -= 0.05
        # if i > 6 * test_env.CTRL_FREQ:
        #     obs[0][2] -= (0.05+0.001*i/test_env.CTRL_FREQ)*np.sin(5*i/test_env.CTRL_FREQ)
        # obs[0][5] -= 0.7
        # if i < 20 * test_env.CTRL_FREQ:
        #     obs[0][0] += 1
        #     obs[0][1] -= 1
        #     obs[0][2] += 0
        #     obs[0][5] += 0.52
        # elif 20 * test_env.CTRL_FREQ < i < 40 * test_env.CTRL_FREQ:
        #     obs[0][0] += 2
        #     obs[0][1] -= 0
        #     obs[0][2] -= 0.5
        #     obs[0][5] += 0.0
        # elif 40 * test_env.CTRL_FREQ < i < 60 * test_env.CTRL_FREQ:
        #     obs[0][0] += 2
        #     obs[0][1] += 2
        #     obs[0][2] -= 1.5
        #     obs[0][5] -= 0.35
        # elif 60 * test_env.CTRL_FREQ < i < 80 * test_env.CTRL_FREQ:
        #     obs[0][0] += 1
        #     obs[0][1] += 3
        #     obs[0][2] -= 0
        #     obs[0][5] -= 0.69
        # elif 80 * test_env.CTRL_FREQ < i:
        #     obs[0][0] += 3
        #     obs[0][1] += 3.5
        #     obs[0][2] -= 1
        #     obs[0][5] += 0


        action, _states = policy.predict(obs,
                                         deterministic=True
                                         )

        print(f"############### PREDICTED ACTION: {action} ####################")

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
        # logger.plot_instantaneous_reward()
        # logger.plot()
        # logger.plot_rpms()
        # logger.plot_trajectory()

    if save:
        logger.save_as_csv(comment)


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
        '--algorithm',
        default='ppo',
        help='The algorithm used for training'
    )
    parser.add_argument(
        '--test_env',
        default=CLStage1Sim2Real,
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

    run_simulation(**vars(parser.parse_args()))
