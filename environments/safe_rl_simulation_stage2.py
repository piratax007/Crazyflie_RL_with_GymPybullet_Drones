import numpy as np
from environments.safe_rl_simulation_stage1 import SafeRLSimulationStage1
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class SafeRLSimulationStage2(SafeRLSimulationStage1):
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
        self.LOG_ANGULAR_VELOCITY = np.zeros((1, 3))
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

    ################################################################################

    @staticmethod
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
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], self.INIT_XYZS[0],
                                          p.getQuaternionFromEuler(self.INIT_RPYS[0]))
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info