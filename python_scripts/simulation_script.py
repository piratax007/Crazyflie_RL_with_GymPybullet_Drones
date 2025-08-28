#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import pybullet as p
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO, SAC, DDPG, TD3
from python_scripts.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

from environments import environment_map

def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(model_class, policy_path, model):
    if os.path.isfile(policy_path + '/' + model):
        return model_class.load(policy_path + '/' + model)

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def helix_trajectory(number_of_points: int = 50, radius: int = 2, angle_range: float = 30.0) -> tuple:
    """
    Generates a set of 3D helix trajectory coordinates and target yaw angles within a specified angle range.
    The helicoidal trajectory is defined by a circular motion in the XY-plane with linear progression in the
    Z direction. The target yaw angles are intended to be follow over the trajectory.

    Args:
        number_of_points (int): Number of points in the trajectory.
        radius (int): Radius of the helix, defining the circular XY-plane motion.
        angle_range (float): Maximum angle range in degrees for yaw oscillation.

    Returns:
        tuple: A tuple containing the following:
            - x_coordinates (numpy.ndarray): x-coordinates of the trajectory.
            - y_coordinates (numpy.ndarray): y-coordinates of the trajectory.
            - z_coordinates (numpy.ndarray): z-coordinates of the trajectory.
            - yaw_angles (numpy.ndarray): Yaw angles of the trajectory with oscillation.

    How to:
    1. Define the initial point for the helix trajectory in the initializer of the test_env:
    ```
    ...
    test_env = test_env(
        initial_xyzs=np.array([[-1.0, 0.0, 0.0]]),
        ...
    ```
    2. Before the simulation loop, call the helix_trajectory function:
    ```
    ...
    x_target, y_target, z_target, yaw_target = helix_trajectory(simulation_seconds, 1)

    for i in range(simulation_seconds):
        ...
    ```
    3. At the beginning of the simulation loop, update the observation vector:
    ```
    ...
    for i in range(simulation_seconds):
        obs[0][0] += x_target[i]
        obs[0][1] += y_target[i]
        obs[0][2] -= z_target[i]
        ...
    ```
    """
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


def lemniscate_trajectory(number_of_points: int = 50, a: float = 2) -> tuple:
    """
    Generates a lemniscate trajectory with specified number of points and scale.
    The function computes the x, y, and z coordinates along a lemniscate curve,
    as well as the yaw angles at each point of the trajectory. The lemniscate
    is scaled by the parameter `a`.

    Args:
        number_of_points: int
            The number of points to compute along the trajectory. Defaults to 50.
        a: float
            The scale of the lemniscate. Defines the size of the curve. Defaults to 2.

    Returns:
        tuple:
            A tuple containing four elements:
            - x_coordinates (numpy.ndarray): The x-coordinates of the trajectory points.
            - y_coordinates (numpy.ndarray): The y-coordinates of the trajectory points.
            - z_coordinates (numpy.ndarray): The z-coordinates (zeros) for the trajectory points.
            - yaw_angles (numpy.ndarray): The yaw angles for the trajectory points.

    How to:
    How to:
    1. Define the initial point for the helix trajectory in the initializer of the test_env:
    ```
    ...
    test_env = test_env(
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        ...
    ```
    2. Before the simulation loop, call the lemniscate_trajectory function:
    ```
    ...
    x_target, y_target, z_target, yaw_target = lemniscate_trajectory(simulation_seconds, 1)

    for i in range(simulation_seconds):
        ...
    ```
    3. At the beginning of the simulation loop, update the observation vector:
    ```
    ...
    for i in range(simulation_seconds):
        obs[0][0] += x_target[i]
        obs[0][1] += y_target[i]
        obs[0][2] -= z_target[i]
        ...
    ```
    """
    t = np.linspace(0, 2.125 * np.pi, number_of_points)
    x_coordinates = a * np.sin(t) / (1 + np.cos(t) ** 2)
    y_coordinates = a * np.sin(t) * np.cos(t) / (1 + np.cos(t) ** 2)
    z_coordinates = np.zeros(number_of_points)

    yaw_angles = np.arctan2(-y_coordinates, -x_coordinates)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles


def smooth_trajectory(points: list, num_points: int = 100):
    """
    Generates a smooth trajectory through a series of 3D points and calculates the corresponding roll,
    pitch, and yaw angles for each point.

    The function takes a list of 3D points, interpolates a smooth trajectory through the points
    using a spline, and computes the tangent vectors at each interpolated point to determine
    the corresponding roll, pitch, and yaw angles. The result is returned as tuples of x, y, z
    coordinates, and yaw angles.

    Arguments:
        points (list): A list of 3D points represented as [x, y, z] coordinates through which the
                       trajectory is interpolated.
        num_points (int): The number of interpolated points generated along the trajectory. Defaults
                          to 100.

    Returns:
        tuple: A tuple containing four elements:
               - A tuple of x-coordinates along the smoothed trajectory.
               - A tuple of y-coordinates along the smoothed trajectory.
               - A tuple of z-coordinates along the smoothed trajectory.
               - A tuple of yaw angles associated with each trajectory point.

    How To:
    1. Before the simulation loop define a list of points (at least four)
    ```
    ...
    points = [
        [0, 0, 0],
        [1, 2, 0.5],
        [2, 4, 1],
        [3, 2, 1.5],
        [4, 0, 2],
        [5, -2, 2.5],
        [6, -4, 3],
        [7, -2, 3.5],
        [8, 0, 4]
    ]

    for i in range(simulation_seconds):
        ...
    ```
    2. After the list of points, and before the simulation loop, call the smooth_trajectory function:
    ```
    ...
    x_target, y_target, z_target, yaw_target = smooth_trajectory(points, simulation_seconds)

    for i in range(simulation_seconds):
        ...
    ```
    3. At the beginning of the simulation loop, update the observation vector:
    ```
    ...
    for i in range(simulation_seconds):
        obs[0][0] += x_target[i]
        obs[0][1] += yaw_target[i]
        obs[0][2] -= z_target[i]
        ...
    ```
    """
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
    """
    Generates random cylindrical positions based on specified constraints.

    This function generates random positions within or outside a cylindrical region.
    The cylinder is defined by its geometry and location in 3D space. Additional
    parameters control the distance and mode of generation. Positions are returned
    as a tuple of x, y, and z coordinates.

    Parameters:
        inner_radius (float): Defines the minimum radius from the cylinder's central
            axis for generating points. Default is 0.0.
        outer_radius (float): Defines the maximum radius from the cylinder's central
            axis for generating points. Default is 1.5.
        cylinder_height (float): Height of the cylinder, defining the range along
            the z-axis. Default is 1.5.
        cylinder_center (tuple): A tuple (cx, cy, cz) representing the center of the
            cylinder. Default is (0, 0, 1).
        mode (str): Specifies the mode of position generation. Can be either "inside"
            to generate points within the cylinder or "outside" for points outside
            the defined cylindrical shell. Default is "inside".
        min_distance (float): The minimum distance outside the cylinder's outer surface
            for the "outside" mode. Default is 0.0.
        max_distance (float): The maximum distance outside the cylinder's outer surface
            for the "outside" mode. Default is 0.0.

    Returns:
        tuple: A tuple containing x, y, and z coordinates of the generated random
        position in 3D space.

    How to:
        Initialize the position of the drone in the initializer of the test_env:
        ```
        ...
        test_env = test_env(
            initial_xyzs = np.array([[*random_cylindrical_positions(outer_radius=2.0, cylinder_height=2, mode='inside')]]),
            ...
        ```
    """
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
    """
    Adds a waypoint in the form of a sphere to the simulation. This function
    creates a sphere visual shape with specified parameters and places it at
    the given position.

    Parameters:
        position (Sequence[float]): A 3D position [x, y, z] where the sphere will
            be placed.
        radius (float, optional): Radius of the sphere to represent the waypoint.
            Defaults to 0.1.
        color (Sequence[float], optional): RGBA color of the sphere. Defaults to
            (1, 0, 0, 0.5).

    Returns:
        int: The ID of the created sphere in the simulation.

    Raises:
        Exception: If the simulation environment is not set up properly.

    How to:
        Before the simulation loop, call the add_way_point function once for each waypoint:
        ```
        ...
        _ = {
            0: add_way_point((-1, 1, 0), radius=0.025),
            1: add_way_point((-1, 1, 1), radius=0.025),
            2: add_way_point((-2, 0, 1.5), radius=0.025)
        }

        for i in range(simulation_seconds):
            ...
        ```
    """
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
        initial_xyzs=np.array([[0.0, 0.0, 0.0]]),
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
        clipped_actions, _states = policy.predict(obs,
                                         deterministic=True
                                         )

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
        default='CLStage1Sim2Real',
        type=str,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--simulation-length',
        default=20,
        type=int,
        help='The length of the simulation in seconds'
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
        record_video=False,
        simulation_length=args.simulation_length,
        reset=args.reset,
        save=args.save,
        comment=args.comment,
        plot=args.plot,
        debug=args.debug,
    )
