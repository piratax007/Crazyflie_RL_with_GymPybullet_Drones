import numpy as np
from gymnasium import spaces
from environments.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class SafeRLSimulationStage1(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 target_rpys=np.array([[0, 0, 0]]),
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
        self.LOG_Z_DISTANCE = np.zeros((1, 1))
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

    def _target_reward(self, state):
        print(f"############## TARGET ORIENTATION {self.TARGET_ORIENTATION} ##############")
        return np.linalg.norm(self.TARGET_POS - state[0:3]) + 1.5*np.linalg.norm(self.TARGET_ORIENTATION[0] - state[
            7:10])

    @staticmethod
    def _update_last(storage: np.ndarray, data):
        if np.shape(storage)[0] == 2:
            storage = np.delete(storage, 0, axis=0)

        return np.vstack((storage, data))

    def _exploration_reward(self, state):
        self.LOG_Z_DISTANCE = self._update_last(self.LOG_Z_DISTANCE, np.linalg.norm(state[0:3] - self.TARGET_POS))
        ret = self.LOG_Z_DISTANCE[1][0] > self.LOG_Z_DISTANCE[0][0] + 0.025
        return ret

    def _stability_reward(self, state):
        if state[7] ** 2 + state[8] ** 2 < 0.001:
            return 2 * np.exp(-np.linalg.norm(self.TARGET_POS - state[0:3]))

        return -(state[7] ** 2 + state[8] ** 2)

    def _get_we_differences(self, state):
        angular_velocities = state[13:16]
        self.LOG_ANGULAR_VELOCITY = self._update_last(self.LOG_ANGULAR_VELOCITY, angular_velocities)
        differences = {
            'roll': self.LOG_ANGULAR_VELOCITY[0][0] - self.LOG_ANGULAR_VELOCITY[1][0],
            'pitch': self.LOG_ANGULAR_VELOCITY[0][1] - self.LOG_ANGULAR_VELOCITY[1][1],
            'yaw': self.LOG_ANGULAR_VELOCITY[0][2] - self.LOG_ANGULAR_VELOCITY[1][2],
        }
        return differences

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        we_differences = self._get_we_differences(state)
        ret = (0.25 - 0.20 * self._target_reward(state) -
               1 * (1 if self._exploration_reward(state) else -0.2) +
               0.20 * self._stability_reward(state) -
               0.18 * (we_differences['roll'] ** 2 + we_differences['pitch'] ** 2 + we_differences['yaw'] ** 2))
        return ret

    ################################################################################

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < .01 and state[7] ** 2 + state[8] ** 2 < 0.001:
            return True

        return False

    ################################################################################

    def _computeTruncated(self):
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
                obs[7:10] + np.random.normal(0.0, 0.002, 3),
                obs[10:13] + np.random.normal(0.0, 0.001, 3),
                obs[13:16] + np.random.normal(0.0, 0.002, 3),
            ]).reshape(12, )
        ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        return ret
