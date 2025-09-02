import numpy as np
from environments.CL_Stage1_S2R_e2e_dr import CLStage1Sim2RealDomainRandomization
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class CLStage2Sim2RealDomainRandomization(CLStage1Sim2RealDomainRandomization):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 target_rpys = np.array([[0, 0, 0]]),
                 physics: Physics = Physics.PYB,
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

    def _target_error(self, state):
        return (np.linalg.norm(self.TARGET_POS - state[0:3]) +
                np.linalg.norm(self.TARGET_ORIENTATION - state[7:10]))

    def _is_away_from_exploration_area(self, state):
        return (np.linalg.norm(state[0:2] - self.TARGET_POS[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + 0.025 or
                state[2] > self.TARGET_POS[2] + 0.025)

    def _is_closed(self, state):
        return np.linalg.norm(state[0:3] - self.TARGET_POS[0:3]) < 0.025

    def _performance(self, state):
        if self._is_closed(state) and state[7] ** 2 + state[8] ** 2 < 0.001:
            return 2

        return -(state[7] ** 2 + state[8] ** 2)

    def _get_previous_current_we(self, current_state):
        if np.shape(self.LOG_ANGULAR_VELOCITY)[0] > 2:
            self.LOG_ANGULAR_VELOCITY = np.delete(self.LOG_ANGULAR_VELOCITY, 0, axis=0)

        return np.vstack((self.LOG_ANGULAR_VELOCITY, current_state[13:16]))

    def _get_we_differences(self, state):
        log = self._get_previous_current_we(state)
        differences = {
            'roll': log[0][0] - log[1][0],
            'pitch': log[0][1] - log[1][1],
            'yaw': log[0][2] - log[1][2],
        }
        return differences

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        we_differences = self._get_we_differences(state)
        ret = (0.25 - 0.20 * self._target_error(state) -
               1 * (1 if self._is_away_from_exploration_area(state) else -0.2) +
               0.20 * self._performance(state) -
               0.18 * (we_differences['roll'] ** 2 + we_differences['pitch'] ** 2 + we_differences['yaw'] ** 2))
        return ret

    ################################################################################

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .02 and state[7] ** 2 + state[8] ** 2 < 0.001:
            return True

        return False

    ################################################################################

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if (np.linalg.norm(state[0:2] - self.TARGET_POS[0:2]) >
                np.linalg.norm(self.INIT_XYZS[0][0:2] - self.TARGET_POS[0:2]) + .05 or
                state[2] > self.TARGET_POS[2] + .05 or
                abs(state[7]) > .25 or abs(state[8]) > .25):
            return True

        if self.step_counter / self.PYB_FREQ > self.EPISODE_LENGTH_SECONDS:
            return True

        return False

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
            options: dict = None,
    ):
        obs, info = super().reset(seed=seed, options=options)

        dr_params = info.get("dr_params", None)

        self.INIT_XYZS = np.array(
            [[*self._random_cylindrical_positions(outer_radius=2.0, cylinder_height=2, mode='inside')]])
        p.resetBasePositionAndOrientation(
            self.DRONE_IDS[0],
            self.INIT_XYZS[0],
            p.getQuaternionFromEuler(self.INIT_RPYS[0]),
            physicsClientId=self.CLIENT
        )

        self._updateAndStoreKinematicInformation()
        obs_new = self._computeObs()
        info_new = self._computeInfo()

        if dr_params is not None:
            info_new["dr_params"] = dr_params

        for k, v in info.items():
            if k not in info_new:
                info_new[k] = v

        return obs_new, info_new