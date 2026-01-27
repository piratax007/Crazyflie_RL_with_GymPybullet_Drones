import numpy as np
from gymnasium import spaces
from environments.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import pybullet as p


class MED26Quaterion(BaseRLAviary):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=np.array([[0, 0, 0.1]]),
                 initial_rpys=np.array([[0, 0, 0]]),
                 target_xyzs=np.array([0, 0, 1]),
                 target_q_xyzw=np.array([0.0, 0.0, 0.5, 0.86602540378443386]),
                 physics: Physics = Physics.PYB_GND,
                 pyb_freq: int = 200,
                 ctrl_freq: int = 100,
                 gui=False,
                 record=False,
                 observation_space: ObservationType = ObservationType.KIN,
                 action_space: ActionType = ActionType.RPM,
                 *arg, **kwargs
                 ):
        self.INIT_XYZS = initial_xyzs
        self.TARGET_POS = target_xyzs
        self.TARGET_QUATERNION = target_q_xyzw
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
                         act=action_space,
                         *arg, **kwargs
                         )

    ################################################################################

    @staticmethod
    def _exponential_reward(b: float, current, target) -> float:
        return np.exp(-b * np.linalg.norm(current - target)**2)

    def _xy_error_reward(self, xy_current, xy_target):
        return self._exponential_reward(0.2, xy_current, xy_target)

    def _z_error_reward(self, z_current, z_target):
        return self._exponential_reward(1, z_current, z_target)

    def _linear_velocity_error_reward(self, v_current, v_target):
        return self._exponential_reward(0.2, v_current, v_target)

    @staticmethod
    def _orientation_error_reward(theta: float) -> float:
        return np.exp(-theta**2)

    def quat_geodesic_angle_from_qerr_xyzw(self, q: np.ndarray) -> float:
        q_err = self._quat_error_xyzw(self.TARGET_QUATERNION, q, ensure_pos_w=True)
        q_err = q_err / np.linalg.norm(q_err)
        v = q_err[:3]
        w = q_err[3]
        w = float(np.clip(w, 0.0, 1.0))
        return float(2.0 * np.arctan2(np.linalg.norm(v), w))

    def _delta_action_penalty(self, drone_id: int = 0, weight: float = 0.01) -> float:
        if len(self.action_buffer) < 2:
            return 0.0

        u_t = np.asarray(self.action_buffer[-1][drone_id, :], dtype=np.float32)
        u_tm1 = np.asarray(self.action_buffer[-2][drone_id, :], dtype=np.float32)

        du = u_t - u_tm1
        return float(weight * np.dot(du, du))

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        xy = state[0:2]
        z = state[2]
        q = state[3:7]
        theta = self.quat_geodesic_angle_from_qerr_xyzw(q)
        v = state[10:13]
        smooth_penalty = self._delta_action_penalty(0, 0.001)
        ret = (0.25
               + 0.17 * self._xy_error_reward(xy, self.TARGET_POS[0:2])
               + 0.18 * self._z_error_reward(z, self.TARGET_POS[2])
               + 0.1 * self._linear_velocity_error_reward(v, np.array([0, 0, 0]))
               + 0.3 * self._orientation_error_reward(theta)
               - smooth_penalty
               )
        return ret

    ################################################################################

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)

        current_position = state[0:3]
        position_error = np.linalg.norm(current_position - self.TARGET_POS)
        current_quaternion = state[3:7]
        current_velocity = state[10:13]
        velocity_norm = np.linalg.norm(current_velocity)
        current_omega = state[13:16]
        omega_norm = np.linalg.norm(current_omega)

        theta = self.quat_geodesic_angle_from_qerr_xyzw(current_quaternion)

        failure = (
        (state[2] < 0.1) or
        (position_error > 3.0) or
        (theta > np.pi) or
        (velocity_norm > 8.0) or
        (omega_norm > 10.0)
        )

        if failure:
            return True

        success = (
        (position_error < 0.25) and
        (theta < 0.01)
        )

        if success:
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
        obs_lower_bound = np.array([[lo, lo, lo, lo ,lo ,lo ,lo , lo, lo, lo, lo, lo, lo, -1, -1, -1, -1]])
        obs_upper_bound = np.array([[hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, hi, 1, 1, 1, 1]])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    @staticmethod
    def _compute_raw_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return current - target

    def _compute_noisy_error(
            self,
            current: np.ndarray,
            target: np.ndarray,
            noise_definition: tuple = (0.0, 0.001, 3)
    ) -> np.ndarray:

        return (self._compute_raw_error(current, target) +
                np.random.normal(noise_definition[0], noise_definition[1], noise_definition[2]))

    @staticmethod
    def _quat_xyzw_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if n < eps:
            raise ValueError("Quat norm is near zero, can't normalize")
        return q / n

    @staticmethod
    def _quat_xyzw_conjugate(q: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
        return  np.array([-x, -y, -z, w], dtype=np.float64)

    @staticmethod
    def _quat_xyzw_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([x, y, z, w], dtype=np.float64)

    @classmethod
    def _quat_error_xyzw(cls, q_target: np.ndarray, q_current: np.ndarray, ensure_pos_w: bool = True) -> np.ndarray:
        q_target = cls._quat_xyzw_normalize(q_target)
        q_current = cls._quat_xyzw_normalize(q_current)
        q_target_inv = cls._quat_xyzw_conjugate(q_target)
        q_e = cls._quat_xyzw_multiply(q_target_inv, q_current)
        if ensure_pos_w and q_e[3] < 0.0:
            q_e *= -1

        return cls._quat_xyzw_normalize(q_e)

    def _noisy_quaternion(self, q: np.ndarray, noise: tuple = (0.0, 0.002, 4)) -> np.ndarray:
        noisy_q = q + np.random.normal(*noise)
        return self._quat_xyzw_normalize(noisy_q)

    def _computeObs(self):
        obs_17 = np.zeros((self.NUM_DRONES, 17))
        for i in range(self.NUM_DRONES):
            obs = self._getDroneStateVector(i)
            q_err = self._quat_error_xyzw(self.TARGET_QUATERNION, obs[3:7], ensure_pos_w=True)
            obs_17[i, :] = np.hstack([
                self._compute_noisy_error(obs[0:3], self.TARGET_POS, (0.0, 0.001, 3)),
                self._noisy_quaternion(q_err, (0.0, 0.002, 4)),
                obs[10:13] + np.random.normal(0.0, 0.001, 3),
                obs[13:16] + np.random.normal(0.0, 0.002, 3),
                np.asarray(self.action_buffer[-1][i, :], dtype=np.float32)
                ]).reshape(17, )
        ret = np.array([obs_17[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
        return ret

    def reset(
            self,
            seed: int = None,
            options: dict = None,
    ):
        p.resetSimulation(physicsClientId=self.CLIENT)
        self._housekeeping()
        self.INIT_XYZS = np.array(
                [
                    [
                        np.random.uniform(-1, 1 + 1e-10, 1)[0],
                        np.random.uniform(-1, 1 + 1e-10, 1)[0],
                        np.random.uniform(0.1, 2 + 1e-10, 1)[0]
                    ]
                ]
            )
        self.INIT_RPYS = np.array([[
            np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
            np.random.uniform(-0.2, 0.2 + 1e-10, 1)[0],
            np.random.uniform(-3.14, 3.14 + 1e-10, 1)[0]
        ]])
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], self.INIT_XYZS[0, :],
                                          p.getQuaternionFromEuler(self.INIT_RPYS[0]),
                                          physicsClientId=self.CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[0], [0, 0, 0], [0, 0, 0], physicsClientId=self.CLIENT)
        self._updateAndStoreKinematicInformation()
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
