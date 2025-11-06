import numpy as np
from environments.safe_rl_simulation_stage2 import SafeRLSimulationStage2
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class SafeRLSimulationStage3(SafeRLSimulationStage2):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 target_rpys = np.array([[0, 0, 0]]),
                 physics: Physics = Physics.PYB_GND,
                 pyb_freq: int = 200,
                 ctrl_freq: int = 100,
                 gui=False,
                 record=False,
                 observation_space: ObservationType = ObservationType.KIN,
                 action_space: ActionType = ActionType.RPM
                 ):
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POS = target_xyzs
        self.TARGET_ORIENTATION = target_rpys
        self.EPISODE_LENGTH_SECONDS = 5
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         observation_space=observation_space,
                         action_space=action_space
                         )

    def reset(
            self,
            seed: int = None,
            option: dict = None,
    ):
        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self._updateAndStoreKinematicInformation()
        self.INIT_XYZS = np.array(
            [[*self._random_cylindrical_positions(outer_radius=2.0, cylinder_height=2, mode='inside')]])
        self.INIT_RPYS = np.array([[
            np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
            np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
            np.random.uniform(-3.14, 3.14 + 1e-10, 1)[0]
        ]])
        initial_linear_velocity = [
            np.random.uniform(-1, 1 + 1e-10, 1)[0],
            np.random.uniform(-1, 1 + 1e-10, 1)[0],
            np.random.uniform(-1, 1 + 1e-10, 1)[0]
        ]
        initial_angular_velocity = [
            np.random.uniform(-1, 1 + 1e-10, 1)[0],
            np.random.uniform(-1, 1 + 1e-10, 1)[0],
            np.random.uniform(-1, 1 + 1e-10, 1)[0]
        ]
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], self.INIT_XYZS[0],
                                          p.getQuaternionFromEuler(self.INIT_RPYS[0]))
        p.resetBaseVelocity(self.DRONE_IDS[0], initial_linear_velocity, initial_angular_velocity)
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info