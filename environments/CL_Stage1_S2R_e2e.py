import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class CLStage1Sim2Real(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 400,
                 ctrl_freq: int = 200,
                 gui=False,
                 record=False,
                 observation_space: ObservationType = ObservationType.KIN,
                 action_space: ActionType = ActionType.ONE_D_RPM
                 ):
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POS = target_xyzs
        self.EPISODE_LENGTH_SECONDS = 5
        self.LOG_ANGULAR_VELOCITY = np.zeros((1, 3))
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=observation_space,
                         act=action_space
                         )

    ################################################################################

    def _target_error(self, state):
        return np.linalg.norm(self.TARGET_POS - state[0:3])

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

    ################################################################################

    def _computeInfo(self):
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def _observationSpace(self):
        lo = -np.inf
        hi = np.inf
        obs_lower_bound = np.array([[lo, lo, 0, lo, lo, lo, lo, lo, lo, lo, lo, lo]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        obs_12 = np.zeros((self.NUM_DRONES, 12))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            obs_12[i, :] = np.hstack([
                obs[0:3] + np.random.normal(0.0, 0.001, 3),
                obs[7:10] + np.random.normal(0.0, 0.001, 3),
                obs[10:13] + np.random.normal(0.0, 0.002, 3),
                obs[13:16] + np.random.normal(0.0, 0.002, 3),
            ]).reshape(12, )
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        return ret
