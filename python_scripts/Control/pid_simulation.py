import time
import argparse
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from python_scripts.Control.PIDControl import PIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

def run(
        drone=DroneModel("cf2x"),
        num_drones=1,
        physics=Physics("pyb_gnd"),
        gui=True,
        record_video=False,
        plot=False,
        user_debug_gui=False,
        obstacles=False,
        simulation_freq_hz=400,
        control_freq_hz=200,
        duration_sec=10,
        output_folder='results',
        colab=False,
        save=False,
        comment=""
        ):

    INIT_XYZS = np.array([[-1, 1, 0]])
    INIT_RPYS = np.array([[0, 0, 0.78]])
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

    ctrl = [PIDControl(drone_model=drone)]

    p.resetDebugVisualizerCamera(1, 125, -10, [1, 1, 1])

    action = np.zeros((1, 4))
    START = time.time()
    for i in range(int(duration_sec * env.CTRL_FREQ)):
        obs, _, terminated, truncated, info = env.step(action)

        action[0, :], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy = target_rpy
        )
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([target_pos, INIT_RPYS[0, :], np.zeros(6)]),
            reward=0
        )
        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()

    if plot:
        logger.plot_position_and_orientation()
        logger.plot_rpms()

    if save:
        logger.save_as_csv(comment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
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
