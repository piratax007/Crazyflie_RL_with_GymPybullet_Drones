import time
import argparse
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from python_scripts.PIDControl import PIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

def _random_cylindrical_positions(
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

def run(
        drone=DroneModel("cf2x"),
        num_drones=1,
        physics=Physics("pyb_gnd"),
        gui=True,
        record_video=False,
        plot=False,
        user_debug_gui=False,
        obstacles=False,
        simulation_freq_hz=240,
        control_freq_hz=240,
        duration_sec=10,
        output_folder='results',
        colab=False,
        save=False,
        comment=""
        ):

    INIT_XYZS = np.array(
        [
            [
                *_random_cylindrical_positions(
                    outer_radius=2.0, cylinder_height=2, mode='inside'
                )
            ]
        ]
    )
    INIT_RPYS = np.array([[
        np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
        np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
        np.random.uniform(-1.57, 1.57 + 1e-10, 1)[0]
    ]])
    target_pos = np.array([0.0, 0.0, 1.0])
    target_rpy = np.array([0.0, 0.0, 0.0])

    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    ctrl = [PIDControl(drone_model=drone) for _ in range(num_drones)]

    action = np.zeros((num_drones,4))
    START = time.time()

    for i in range(int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=target_pos,
                target_rpy = target_rpy
            )
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([target_pos, INIT_RPYS[j, :], np.zeros(6)]),
                reward=0
            )
            env.render()
            if gui:
                sync(i, START, env.CTRL_TIMESTEP)

    env.close()

    if plot:
        logger.plot_position_and_orientation()

    if save:
        logger.save_as_csv(comment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument(
        '--num_drones',
        default=1,
        type=int,
        help='Number of drones (default: 3)',
        metavar=''
    )
    parser.add_argument(
        '--duration_sec',
        default=10,
        type=int,
        help='Duration of the simulation in seconds (default: 5)',
        metavar=''
    )
    parser.add_argument(
        '--output_folder',
        default='results',
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar=''
    )
    parser.add_argument(
        '--save',
        default=False,
        type=str2bool,
        help='Whether to save the simulation results (default: "False")',
        metavar=''
    )
    parser.add_argument(
        '--comment',
        default='',
        type=str,
        help='Comment for the log file (default: "")',
        metavar=''
    )
    parser.add_argument(
        '--plot',
        default=False,
        type=str2bool,
        help='Whether to plot the simulation results (default: "False)',
        metavar=''
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
