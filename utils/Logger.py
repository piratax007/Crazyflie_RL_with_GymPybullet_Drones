import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str = "results",
                 num_drones: int = 1,
                 duration_sec: int = 0,
                 colab: bool = False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.COLAB = colab
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec * self.LOGGING_FREQ_HZ))
        # Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_drones, 16, duration_sec * self.LOGGING_FREQ_HZ))  # 16 states: pos_x,
        # pos_y,
        # pos_z,
        # vel_x,
        # vel_y,
        # vel_z,
        # roll,
        # pitch,
        # yaw,
        # ang_vel_x,
        # ang_vel_y,
        # ang_vel_z,
        # rpm0,
        # rpm1,
        # rpm2,
        # rpm3
        # Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_drones, 12, duration_sec * self.LOGGING_FREQ_HZ))  # 12 control targets: pos_x,
        # pos_y,
        # pos_z,
        # vel_x,
        # vel_y,
        # vel_z,
        # roll,
        # pitch,
        # yaw,
        # ang_vel_x,
        # ang_vel_y,
        # ang_vel_z
        self.rewards = np.zeros((num_drones, duration_sec * self.LOGGING_FREQ_HZ))

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            reward,
            control=np.zeros(12)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        reward : float
            Instantaneous reward of the drone.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        # Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
            self.rewards = np.concatenate((self.rewards, np.zeros((self.NUM_DRONES, 1))), axis=1)
        # Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1] - 1
        # Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        # Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1
        self.rewards[drone, current_counter] = reward

    ################################################################################

    def save(self, comment: str = "") -> None:
        """Save the logs to file.
        Parameters
        ----------
        comment : str, optional
            Added to the foldername.
        """
        with (open(os.path.join(self.OUTPUT_FOLDER,
                                "save-flight-" + comment + "-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S") + ".npy"),
                   'wb')
              as out_file):
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str = ""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        csv_dir = os.path.join(self.OUTPUT_FOLDER,
                               "save-flight-" + comment + "-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir + '/')
        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)
        for i in range(self.NUM_DRONES):
            with open(csv_dir + "/x" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0, :]])), delimiter=",")
            with open(csv_dir + "/y" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 1, :]])), delimiter=",")
            with open(csv_dir + "/z" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 2, :]])), delimiter=",")
            ####
            with open(csv_dir + "/r" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, list(map(lambda angle: (angle * 180) / np.pi, self.states[i, 6, :]))])), delimiter=",")
            with open(csv_dir + "/p" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, list(map(lambda angle: (angle * 180) / np.pi, self.states[i, 7, :]))])), delimiter=",")
            with open(csv_dir + "/ya" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, list(map(lambda angle: (angle * 180) / np.pi, self.states[i, 8, :]))])), delimiter=",")
            ####
            with open(csv_dir + "/rr" + str(i) + ".csv", 'wb') as out_file:
                rdot = np.hstack([0, (self.states[i, 6, 1:] - self.states[i, 6, 0:-1]) * self.LOGGING_FREQ_HZ])
                np.savetxt(out_file, np.transpose(np.vstack([t, rdot])), delimiter=",")
            with open(csv_dir + "/pr" + str(i) + ".csv", 'wb') as out_file:
                pdot = np.hstack([0, (self.states[i, 7, 1:] - self.states[i, 7, 0:-1]) * self.LOGGING_FREQ_HZ])
                np.savetxt(out_file, np.transpose(np.vstack([t, pdot])), delimiter=",")
            with open(csv_dir + "/yar" + str(i) + ".csv", 'wb') as out_file:
                ydot = np.hstack([0, (self.states[i, 8, 1:] - self.states[i, 8, 0:-1]) * self.LOGGING_FREQ_HZ])
                np.savetxt(out_file, np.transpose(np.vstack([t, ydot])), delimiter=",")
            ###
            with open(csv_dir + "/vx" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3, :]])), delimiter=",")
            with open(csv_dir + "/vy" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 4, :]])), delimiter=",")
            with open(csv_dir + "/vz" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 5, :]])), delimiter=",")
            ####
            with open(csv_dir + "/wx" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9, :]])), delimiter=",")
            with open(csv_dir + "/wy" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 10, :]])), delimiter=",")
            with open(csv_dir + "/wz" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 11, :]])), delimiter=",")
            ####
            with open(csv_dir + "/rpm0-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12, :]])), delimiter=",")
            with open(csv_dir + "/rpm1-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 13, :]])), delimiter=",")
            with open(csv_dir + "/rpm2-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 14, :]])), delimiter=",")
            with open(csv_dir + "/rpm3-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 15, :]])), delimiter=",")
            ####
            with open(csv_dir + "/pwm0-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 12, :] - 4070.3) / 0.2685])),
                           delimiter=",")
            with open(csv_dir + "/pwm1-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 13, :] - 4070.3) / 0.2685])),
                           delimiter=",")
            with open(csv_dir + "/pwm2-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 14, :] - 4070.3) / 0.2685])),
                           delimiter=",")
            with open(csv_dir + "/pwm3-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 15, :] - 4070.3) / 0.2685])),
                           delimiter=",")
            with open(csv_dir + "/reward-" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.rewards[i]])))

    ################################################################################

    def plot(self, pwm=False):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        # Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)

        col = 0

        row = 0
        # X
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        # ROLL
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('roll (rad)')

        row = 2
        # Y
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')

        row = 3
        # PITCH
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pitch (rad)')

        row = 4
        # Z
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')

        row = 5
        # YAW
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('yaw (rad)')

        # Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')

        # Time ##################################################
        row = 9
        axs[row, col].plot(t, t, label="time")
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('time')

        col = 1

        # Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')

        # RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            rdot = np.hstack([0, (self.states[j, 6, 1:] - self.states[j, 6, 0:-1]) * self.LOGGING_FREQ_HZ])
            axs[row, col].plot(t, rdot, label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('rdot (rad/s)')
        row = 4
        for j in range(self.NUM_DRONES):
            pdot = np.hstack([0, (self.states[j, 7, 1:] - self.states[j, 7, 0:-1]) * self.LOGGING_FREQ_HZ])
            axs[row, col].plot(t, pdot, label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        row = 5
        for j in range(self.NUM_DRONES):
            ydot = np.hstack([0, (self.states[j, 8, 1:] - self.states[j, 8, 0:-1]) * self.LOGGING_FREQ_HZ])
            axs[row, col].plot(t, ydot, label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')

        # This IF converts RPM into PWM for all drones ###########
        # except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12, 16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        # RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_" + str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        # Drawing options #######################################
        for i in range(10):
            for j in range(2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                                 frameon=True
                                 )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )
        if self.COLAB:
            plt.savefig(os.path.join('results', 'output_figure.png'))
        else:
            plt.show()

    def plot_instantaneous_reward(self):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        log_instantaneous_reward: a list with the historic instantaneous reward
        """
        font = {'family': 'serif', 'weight': 'normal', 'size': 22}
        plt.rc('font', **font)

        fig, axs = plt.subplots(1)

        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)

        axs.plot(t, self.rewards[0, :], color="tab:blue")
        axs.set_xlabel('time')
        axs.set_ylabel('instantaneous reward')
        axs.grid(True)

        plt.show()

    def plot_trajectory(self):
        font = {'family': 'serif', 'weight': 'normal', 'size': 15}
        plt.rc('font', **font)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot(self.states[0, 0, :], self.states[0, 1, :], self.states[0, 2, :])

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-0, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=20., azim=-35, roll=0)

        plt.show()

    def plot_position_and_orientation(self):
        """Logs entries for a single simulation step, of a single drone."""

        font = {'family': 'serif', 'weight': 'bold', 'size': 22}
        plt.rc('font', **font)
        plt.rcParams['text.usetex'] = True

        fig, axs = plt.subplots(3, 2)

        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)

        col = 0

        row = 0
        # X
        axs[row, col].plot(t, self.states[0, 0], color="tab:blue", linewidth=2.5)
        # axs[row, col].set_yticklabels(axs[row, col].get_yticks(), weight='bold')
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')
        axs[row, col].set_ylim([-1, 1])
        axs[row, col].grid(True)

        row = 1
        # Y
        axs[row, col].plot(t, self.states[0, 1], color="tab:orange", linewidth=2.5)
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')
        axs[row, col].set_ylim([-1, 1])
        axs[row, col].grid(True)

        row = 2
        # Z
        axs[row, col].plot(t, self.states[0, 2], color="tab:green", linewidth=2.5)
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')
        axs[row, col].grid(True)

        col = 1

        row = 0
        # ROLL
        axs[row, col].plot(t, (self.states[0, 6] * 180) / np.pi, color="tab:blue", linewidth=2)
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel(r"$\phi^\circ$")
        axs[row, col].set_ylim([-8.6, 8.6])
        axs[row, col].grid(True)

        row = 1
        # PITCH
        axs[row, col].plot(t, (self.states[0, 7] * 180) / np.pi, color="tab:orange", linewidth=2)
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel(r"$\theta^\circ$")
        axs[row, col].set_ylim([-8.6, 8.6])
        axs[row, col].grid(True)

        row = 2
        # YAW
        axs[row, col].plot(t, (self.states[0, 8] * 180) / np.pi, color="tab:green", linewidth=2)
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel(r"$\psi^\circ$")
        axs[row, col].grid(True)

        fig.subplots_adjust(left=0.08,
                            bottom=0.08,
                            right=0.98,
                            top=0.92,
                            wspace=0.15,
                            hspace=0.0
                            )

        plt.show()

    def plot_rpms(self):
        font = {'family': 'serif', 'weight': 'normal', 'size': 25}
        plt.rc('font', **font)

        fig, axs = plt.subplots(2, 2)

        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)

        col = 0

        row = 0
        axs[row, col].plot(t, self.states[0, 12], label="drone_" + str(0))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('RPM_0')

        row = 1
        axs[row, col].plot(t, self.states[0, 13], label="drone_" + str(0))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('RPM_1')

        col = 1
        row = 0
        axs[row, col].plot(t, self.states[0, 14], label="drone_" + str(0))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('RPM_2')

        row = 1
        axs[row, col].plot(t, self.states[0, 15], label="drone_" + str(0))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('RPM_3')

        plt.show()

    def plot_angular_velocities(self):
        font = {'family': 'serif', 'weight': 'normal', 'size': 22}
        plt.rc('font', **font)
        plt.rcParams['text.usetex'] = True

        fig, axs = plt.subplots(3, 1)

        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)

        row = 0
        rdot = np.hstack([0, (self.states[0, 6, 1:] - self.states[0, 6, 0:-1]) * self.LOGGING_FREQ_HZ])
        axs[row].plot(t, rdot, color="olivedrab")
        axs[row].set_ylabel(r"$\dot{\phi}$ (rad/s)")
        axs[row].grid(True)
        row = 1
        pdot = np.hstack([0, (self.states[0, 7, 1:] - self.states[0, 7, 0:-1]) * self.LOGGING_FREQ_HZ])
        axs[row].plot(t, pdot, color="olivedrab")
        axs[row].set_ylabel(r"$\dot{\theta}$ (rad/s)")
        axs[row].grid(True)
        row = 2
        ydot = np.hstack([0, (self.states[0, 8, 1:] - self.states[0, 8, 0:-1]) * self.LOGGING_FREQ_HZ])
        axs[row].plot(t, ydot, color="olivedrab")
        axs[row].set_xlabel('time')
        axs[row].set_ylabel(r"$\dot{\psi}$ (rad/s)")
        axs[row].grid(True)

        plt.show()
