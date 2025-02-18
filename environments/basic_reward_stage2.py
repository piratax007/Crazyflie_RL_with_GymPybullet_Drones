import gymnasium.spaces.box
import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class BasicRewardStage2(BaseRLAviary):
    def __init__(
            self,
            drone_model: DroneModel = DroneModel.CF2X,
            initial_xyzs = np.array([[0, 0, 0]]),
            initial_rpys = np.array([[0, 0, 0]]),
            target_xyzs = np.array([0, 0, 1]),
            physics: Physics = Physics.PYB_GND,
            pybullet_frequency: int = 240,
            ctrl_freq: int = 30,
            gui = False,
            record = False,
            observation_space: ObservationType = ObservationType.KIN,
            action_space: ActionType = ActionType.RPM,
    ):
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POSITION = target_xyzs
        self.EPISODE_LENGTH_SECONDS = 5
        self.LOG_ANGULAR_VELOCITY = np.zeros((1, 3))
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            pyb_freq=pybullet_frequency,
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=record,
            obs=observation_space,
            act=action_space
        )

    def _target_error(self, state: np.ndarray) -> np.floating:
        return np.linalg.norm(self.TARGET_POSITION - state[0:3])

    def _computeReward(self) -> float:
        state = self._getDroneStateVector(0)
        ret = 25 - 20 * self._target_error(state)
        return ret

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

    def _computeTerminated(self) -> bool:
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POSITION - state[0:3]) < .025 and state[7]**2 + state[8]**2 < 0.01:
            return True

        return False

    def _computeTruncated(self) -> bool:
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LENGTH_SECONDS:
            return True

        return False

    def _computeInfo(self) -> dict:
        return {"answer": 42}

    def _observationSpace(self) -> gymnasium.spaces.box.Box:
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def _computeObs(self) -> np.ndarray:
        obs_12 = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        return ret

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