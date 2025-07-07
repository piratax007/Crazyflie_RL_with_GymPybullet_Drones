import matplotlib.pyplot as plt
import numpy as np
from utils import euler_to_rotation_matrix


def add_vertical_lines(axes: plt.Axes, x_positions: list, y_min: float = 0, y_max: float = 1, label: str = '') -> None:
    for i in range(len(x_positions)):
        axes.axvline(x_positions[i], y_min, y_max, ls='-.', color='gray', label=label if (i == 0) else '')

    axes.legend()


def add_double_arrow_with_label_2d(
        axes: plt.Axes,
        start: tuple[float, float],
        end: tuple[float, float],
        label: str = '',
        label_position: tuple[float, float] = (0, 0),
        horizontal_offset: float = 0.1,
        vertical_offset: float = 0.1
) -> None:
    axes.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='<->', color='black', linewidth=2))

    # mid_point = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    axes.text(
        label_position[0] - horizontal_offset,
        label_position[1] - vertical_offset,
        label, color='black',
        ha='left',
        va='top'
    )


def add_cylinder(
        axes: plt.Axes,
        radius: float = 2.0,
        height: float = 2.0,
        center: tuple = (0, 0, 1),
        color: str = 'blue',
        alpha: float = 0.25
) -> None:
    z = np.linspace(0, height, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(theta_grid) + center[1]
    z_grid = z_grid + center[2] - height / 2

    axes.plot_surface(x_grid, y_grid, z_grid, alpha=alpha, rstride=5, cstride=5, color=color)


def add_sphere(axes: plt.Axes, center: tuple = (0, 0, 0.978), radius: float = 0.1) -> None:
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2 * np.pi, 100)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    x_grid = radius * np.sin(phi_grid) * np.cos(theta_grid) + center[0]
    y_grid = radius * np.sin(phi_grid) * np.sin(theta_grid) + center[1]
    z_grid = radius * np.cos(phi_grid) + center[2]

    axes.plot_surface(x_grid, y_grid, z_grid, alpha=0.25, rstride=5, cstride=5, color='red')


def add_text(axis: plt.Axes, position: tuple, text: str, color: str = 'black', fontsize: int = 20) -> None:
    axis.text(position[0], position[1], position[2], text, color=color, fontsize=fontsize)


def add_inertial_frame(position: tuple, axes: plt.Axes, label_offset: tuple = (0.1, 0.1, 0.1)) -> None:
    rotation_matrix = euler_to_rotation_matrix((0, 0, 0))

    origin = np.array(position)
    inertial_frame_x = rotation_matrix @ np.array([1, 0, 0])
    inertial_frame_y = rotation_matrix @ np.array([0, 1, 0])
    inertial_frame_z = rotation_matrix @ np.array([0, 0, 1])

    axes.quiver(*origin, *inertial_frame_x, color='red', length=0.5, normalize=True)
    axes.quiver(*origin, *inertial_frame_y, color='green', length=0.5, normalize=True)
    axes.quiver(*origin, *inertial_frame_z, color='blue', length=0.5, normalize=True)

    axes.text(
        position[0] + label_offset[0],
        position[1] + label_offset[1],
        position[2] + label_offset[2],
        r'$\mathcal{I}$',
        color='black',
        fontsize=18
    )


def add_body_frame(positions: tuple, attitudes: tuple, axes: plt.Axes) -> None:
    def arrange(data: tuple):
        arranged_data = []
        for row in range(len(data[0])):
            arranged_data.append((data[0][row], data[1][row], data[2][row]))

        return tuple(arranged_data)

    rotation_matrix = tuple(map(lambda a: euler_to_rotation_matrix(a), arrange(attitudes)))
    arranged_positions = arrange(positions)

    for i in range(len(arranged_positions)):
        origin = np.array(arranged_positions[i])
        body_frame_x = rotation_matrix[i] @ np.array([1, 0, 0])
        body_frame_y = rotation_matrix[i] @ np.array([0, 1, 0])
        body_frame_z = rotation_matrix[i] @ np.array([0, 0, 1])

        axes.quiver(*origin, *body_frame_x, color='red', length=0.1, normalize=True)
        axes.quiver(*origin, *body_frame_y, color='green', length=0.1, normalize=True)
        axes.quiver(*origin, *body_frame_z, color='blue', length=0.1, normalize=True)


# def add_body_frame(position: tuple, attitude: tuple, axes: plt.Axes) -> None:
#     rotation_matrix = euler_to_rotation_matrix(attitude)
#
#     origin = np.array(position)
#     rotated_vectors = rotation_matrix @ np.identity(3)
#
#     axes.quiver(*origin, *rotated_vectors[:, 0], color='red', length=0.25, normalize=True)
#     axes.quiver(*origin, *rotated_vectors[:, 1], color='green', length=0.25, normalize=True)
#     axes.quiver(*origin, *rotated_vectors[:, 2], color='blue', length=0.25, normalize=True)

def add_linear_velocity_vector(body_frame_origin: tuple, velocities: tuple, axes: plt.Axes) -> None:
    origin = np.array(body_frame_origin)
    axes.quiver(*origin, *velocities, color='black', length=0.5, arrow_length_ratio=0.25, linestyle='solid', linewidth=1.5)