import math
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class LeeControl(BaseControl):
    """
    Lee Controller for Crazyflie drones based on the controller_lee.c implementation from Crazyflie firmware.

    This controller implements the geometric tracking control law on SE(3) as described in:

      Taeyoung Lee, Melvin Leok, and N. Harris McClamroch,
      "Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010.

    The controller consists of a position controller and an attitude controller.
    The position controller computes a desired force using a PD+I law with gain limits.
    The attitude controller computes the rotation error via a vee-map and then computes
    the required moment command using proportional, derivative, and integral gains as well
    as a term accounting for the gyroscopic effect (omega x J*omega).

    Finally, thrust and moment commands are mapped to motor commands using the mixer matrix,
    and then converted to PWM and RPM values.

    Gains and limits are set to mirror those defined in the C implementation.
    """
    def __init__(self,
                 drone_model: DroneModel,
                 g: float = 9.8,
                 mass: float = 0.03):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL not in [DroneModel.CF2X, DroneModel.CF2P]:
            print("[ERROR] LeeControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()

        # --- Physical constants and parameters ---
        self.mass = mass
        # Inertia (diagonal elements - kg m^2)
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

        # --- Position control gains and limits ---
        self.Kpos_P = np.array([7.5, 7.5, 2.0])
        self.Kpos_D = np.array([5.2, 5.2, 5.0])
        self.Kpos_I = np.array([3.25, 3.25, 4.5])
        self.Kpos_P_limit = 100.0
        self.Kpos_D_limit = 100.0
        self.Kpos_I_limit = 2.0

        # --- Attitude control gains ---
        # self.KR      = np.array([0.007, 0.007, 0.008])
        # self.Komega  = np.array([0.00115, 0.00115, 0.002])
        # self.KI      = np.array([0.03, 0.03, 0.03])
        self.KR      = np.array([0.025, 0.025, 0.008])
        self.Komega  = np.array([0.00115, 0.00115, 0.002])
        self.KI      = np.array([0.05, 0.05, 0.03])

        # --- Mixer and PWM conversion parameters ---
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([
                [-0.5, -0.5, -1.0],
                [-0.5,  0.5,  1.0],
                [ 0.5,  0.5, -1.0],
                [ 0.5, -0.5,  1.0]
            ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                [0.0, -1.0, -1.0],
                [1.0,  0.0,  1.0],
                [0.0,  1.0, -1.0],
                [-1.0, 0.0,  1.0]
            ])

        # --- Controller integral errors (position and attitude) ---
        self.integral_error_pos = np.zeros(3)
        self.i_error_att = np.zeros(3)

        # A placeholder for the desired rotation matrix (for logging)
        self.R_des = np.eye(3)

        self.control_counter = 0

    def reset(self):
        """Resets the integral errors for both position and attitude controllers."""
        self.integral_error_pos = np.zeros(3)
        self.i_error_att = np.zeros(3)
        self.control_counter = 0

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """
        Computes motor commands using the Lee geometric controller.

        Parameters:
            control_timestep (float): Control timestep.
            cur_pos (ndarray): Current position [x, y, z].
            cur_quat (ndarray): Current orientation as a quaternion [x, y, z, w].
            cur_vel (ndarray): Current linear velocity.
            cur_ang_vel (ndarray): Current angular velocity (in rad/s).
            target_pos (ndarray): Desired position.
            target_rpy (ndarray): Desired roll, pitch, yaw (only yaw is used for heading).
            target_vel (ndarray): Desired velocity.
            target_rpy_rates (ndarray): Desired angular rates (only yaw rate is used).

        Returns:
            rpm (ndarray): Motor RPM commands (4,).
            position_error (ndarray): Position error vector.
            yaw_error (float): Yaw error (desired yaw minus current yaw).
        """
        self.control_counter += 1

        # -------------------- Position Controller --------------------
        desired_acceleration = np.array([0.0, 0.0, self.GRAVITY])
        position_error = target_pos - cur_pos
        position_error = np.clip(position_error, -self.Kpos_P_limit, self.Kpos_P_limit)

        velocity_error = target_vel - cur_vel
        velocity_error = np.clip(velocity_error, -self.Kpos_D_limit, self.Kpos_D_limit)

        # Update and clamp the integral error for position
        self.integral_error_pos = self.integral_error_pos + control_timestep * position_error
        self.integral_error_pos = np.clip(self.integral_error_pos, -self.Kpos_I_limit, self.Kpos_I_limit)

        # Compute the desired force:
        desired_force = (
                desired_acceleration + self.Kpos_D * velocity_error + self.Kpos_P * position_error +
                self.Kpos_I * self.integral_error_pos
        )

        # Obtain the current rotation matrix from the current quaternion.
        R = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        thrustSi = self.mass * np.dot(desired_force, R[:, 2])
        if thrustSi < 0.01:
            self.reset()

        # -------------------- Desired Rotation Matrix --------------------
        # Use the magnitude of thrust as a proxy for whether desired_force is significant.
        if thrustSi > 0:
            z_des = desired_force / np.linalg.norm(desired_force)
        else:
            z_des = np.array([0.0, 0.0, 1.0])

        # Desired yaw (in radians) from target_rpy[2]
        desiredYaw = target_rpy[2]
        # Reference horizontal direction based on desired yaw.
        x_c_des = np.array([math.cos(desiredYaw), math.sin(desiredYaw), 0.0])
        # Compute desired y-axis as cross product of z_des and x_c_des.
        z_cross_x = np.cross(z_des, x_c_des)
        normZX = np.linalg.norm(z_cross_x)
        if normZX > 1e-6:
            y_des = z_cross_x / normZX
        else:
            y_des = np.array([0.0, 1.0, 0.0])
        # Compute desired x-axis to complete the rotation matrix.
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack((x_des, y_des, z_des))
        self.R_des = R_des  # store for logging

        # -------------------- Attitude Controller --------------------
        # Compute rotation error using the vee-map:
        R_des_T_R = np.dot(R_des.T, R)
        R_T_R_des = np.dot(R.T, R_des)
        R_error_mat = R_des_T_R - R_T_R_des
        e_R = 0.5 * np.array([R_error_mat[2, 1],
                              R_error_mat[0, 2],
                              R_error_mat[1, 0]])

        # Current angular velocity (assumed to be in rad/s)
        omega = cur_ang_vel

        # -------------------- Desired Angular Velocity --------------------
        # Assume zero jerk.
        desJerk = np.array([0.0, 0.0, 0.0])
        hw = np.array([0.0, 0.0, 0.0])
        if abs(thrustSi) > 1e-6:
            hw = (self.mass / thrustSi) * (desJerk - np.dot(z_des, desJerk) * z_des)
        # Desired yaw rate: use target_rpy_rates[2] scaled by the projection of z_des onto [0,0,1]
        z_w = np.array([0.0, 0.0, 1.0])
        desiredYawRate = target_rpy_rates[2] * np.dot(z_des, z_w)
        # Form desired angular velocity vector:
        desired_omega = np.array([-np.dot(hw, y_des),
                              np.dot(hw, x_des),
                              desiredYawRate])
        # Map desired angular velocity to the body frame:
        omega_r = np.dot(np.dot(R.T, R_des), desired_omega)

        # Angular velocity error:
        omega_error = omega - omega_r

        # Update and accumulate the integral error for attitude.
        self.i_error_att = self.i_error_att + control_timestep * e_R

        # -------------------- Moment Command --------------------
        # Compute moment command:
        moment = (- self.KR * e_R
                  - self.Komega * omega_error
                  - self.KI * self.i_error_att
                  - np.cross(omega, self.J * omega))

        # -------------------- Motor Command Conversion --------------------
        # Convert thrust to a PWM-equivalent command.
        # Note: self.KF is assumed to be defined in BaseControl.
        thrust_cmd = (math.sqrt(thrustSi / (4 * self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
        # Combine thrust and moment commands using the mixer matrix.
        pwm = thrust_cmd + np.dot(self.MIXER_MATRIX, moment)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        rpm = self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST

        # -------------------- Yaw Error Calculation --------------------
        current_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        yaw_error = desiredYaw - current_rpy[2]

        return rpm, position_error, yaw_error
