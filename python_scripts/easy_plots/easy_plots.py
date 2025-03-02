import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd


class PATH:
    def __init__(self, path):
        self.path = path


GENERAL_PATH = PATH("")


def _get_data_from_csv(file: str) -> tuple:
    data_file = os.path.join(GENERAL_PATH.path, file)

    with open(data_file, 'r') as f:
        data_set = tuple(csv.reader(f, delimiter=','))
        x = tuple(map(lambda row: float(row[0]), data_set))
        y = tuple(map(lambda row: float(row[1]), data_set))
        try:
            z = tuple(map(lambda row: float(row[2]), data_set))
            return x, y, z
        except IndexError:
            return x, y


def _export_tuple_to_csv(data: tuple, path: os.path, file_name: str) -> None:
    array_data = np.array(data)
    with open(path + file_name + ".csv", 'wb') as csv_file:
        np.savetxt(csv_file, np.transpose(array_data), delimiter=",")


def combine_data_from(files: list, save_to_csv: bool = False, path: str = '', file_name: str = '') -> tuple:
    combined_data = tuple(map(lambda file: _get_data_from_csv(file)[1], files))

    if save_to_csv:
        _export_tuple_to_csv(combined_data, path, file_name)

    return combined_data


def _plot_references(
        files: list,
        axis: plt.Axes,
        labeled: bool = True,
        label: str = 'Reference',
        style: str = '--'
) -> None:
    for i, file in enumerate(files):
        reference = _get_data_from_csv(file)
        if style == "point" and len(reference) == 3:
            axis.scatter(reference[0], reference[1], reference[2], color='black', label=label, s=100)
        else:
            axis.plot(
                *reference,
                color='black',
                linestyle=style,
                linewidth=2,
                label=label if (i == 0 and labeled) else None
            )


def _interior_axes(create: bool, axes: plt.Axes, settings: dict) -> any:
    if create:
        interior_axes = axes.inset_axes(
            settings['x_y_width_height'],
            xlim=settings['x_portion'],
            ylim=settings['y_portion'],
        )
        interior_axes.set_xticklabels([])
        interior_axes.set_yticklabels([])

        return interior_axes

    return None


def _parse_references(references: dict) -> dict:
    if not references['show']:
        references['labeled'] = False
        references['label'] = ''
        references['files'] = []

    if references['show'] and not references['labeled']:
        references['label'] = ''

    if 'style' not in references.keys():
        references['style'] = '--'

    if not references['interior_detail']:
        references['interior_detail_settings'] = dict()

    return references


def _traces_from_csv(
        files: list,
        labels: list,
        axis: plt.Axes,
        references: dict,
        line_style: str = '-',
        **colors: dict
) -> None:
    parsed_references = _parse_references(references)

    interior_axes = _interior_axes(
        parsed_references['interior_detail'],
        axis,
        parsed_references['interior_detail_settings']
    )

    for i, file in enumerate(files):
        if colors['color_mode'] == 'custom'  and type(colors['color_list'][i][1]) is float:
            color = colors['color_list'][i][0]
            alpha = colors['color_list'][i][1]
        elif colors['color_mode'] == 'custom':
            color = colors['color_list'][i]
            alpha = 1
        else:
            color = None
            alpha = 1

        data = _get_data_from_csv(file)
        if interior_axes is not None:
            interior_axes.plot(*data, color=color, alpha=alpha)
            axis.indicate_inset_zoom(interior_axes, edgecolor='lightgray', alpha=0.25)
        axis.plot(
            *data,
            color=color,
            label=labels[i] if labels[i] != '' else None,
            linestyle=line_style,
            linewidth=1.5,
            alpha=alpha
        )

    if parsed_references['show']:
        _plot_references(
            parsed_references['files'],
            axis,
            parsed_references['labeled'],
            parsed_references['label'],
            parsed_references['style']
        )

    # axis.legend()
    # 2D
    axis.legend(bbox_to_anchor=(0, 1, 1, 0.25), loc="lower right", borderaxespad=0.5, ncol=4)
    # 3D
    # axis.legend(bbox_to_anchor=(0, 0.85, 1, 0.75), loc="lower left", borderaxespad=0, ncol=4)


def _parse_settings(settings: dict) -> dict:
    if 'axes_format' not in settings:
        settings['axes_format'] = 'plain'

    if 'axes_aspect' not in settings:
        settings['axes_aspect'] = 'auto'

    return settings


def _set_axis(axis: plt.Axes, settings: dict) -> None:
    # ToDo: Improve exceptions
    parsed_settings = _parse_settings(settings)

    if parsed_settings['limits']['mode'] != 'auto':
        axis.set_xlim(parsed_settings['limits']['x_range'])
        axis.set_ylim(parsed_settings['limits']['y_range'])
        try:
            axis.set_zlim(parsed_settings['limits']['z_range'])
        except:
            pass

    axis.set_xlabel(parsed_settings['labels']['x_label'], labelpad=15)
    axis.set_ylabel(parsed_settings['labels']['y_label'], labelpad=15)
    try:
        axis.set_zlabel(parsed_settings['labels']['z_label'], labelpad=20)
    except:
        pass

    axis.set_title(parsed_settings['labels']['title'])
    axis.ticklabel_format(axis='both', style=parsed_settings['axes_format'], scilimits=(0, 0))
    axis.set_aspect(parsed_settings['axes_aspect'])
    # axis.set_box_aspect([6,9,4])


def _add_vertical_lines(axes: plt.Axes, x_positions: list, y_min: float = 0, y_max: float = 1, label: str = '') -> None:
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


def single_axis_2d(files: list, labels: list, references: dict, colors: dict, settings: dict, callbacks: list = None) -> None:
    _, axis = plt.subplots(1)

    _traces_from_csv(files, labels, axis, references, **colors)
    _set_axis(axis, settings)

    if callbacks is not None:
        for callback in callbacks:
            callback(axis)

    # TRAINING METRICS
    # _add_vertical_lines(axis, x_positions=[6000000, 6200000, 7000000, 17950000], y_min=0.0, y_max=1.0, label='Training stopped')
    # _add_vertical_lines(axis, x_positions=[17950000], y_min=0.0, y_max=1.0)
    #  TRANSFERENCE KNOWLEDGE
    # add_double_arrow_with_label_2d(axis, (73, 35), (146, 35), 'Time to Threshold', label_position=(75.4, 35.5), vertical_offset=2.7)
    # add_double_arrow_with_label_2d(axis, (588, 36.86), (588, 42.4), 'Asymptotic Performance', label_position=(375, 39), vertical_offset=-1)
    plt.show()


def _select_equally_spaced_sample(data: tuple, sample_size: int) -> tuple:
    step = len(data[0]) // sample_size
    r = tuple(tuple(inner_tuple[i * step] for i in range(sample_size)) for inner_tuple in data)
    return r


def single_axis_3d(
        files: list,
        labels: list,
        references: dict,
        colors: dict,
        settings: dict,
        decorations: dict = None,
        callbacks: list = None
) -> None:
    plt.rcParams['text.usetex'] = True
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    _traces_from_csv(
        files,
        labels,
        axis,
        references,
        line_style=settings['line_style'] if 'line_style' in settings else '-',
        line_width=settings['line_width'] if 'line_width' in settings else 1.5,
        **colors
    )

    if decorations is not None and decorations['show']:
        for i in range(len(decorations['position_files'])):
            positions = _select_equally_spaced_sample(
                combine_data_from(decorations['position_files'][i]),
                decorations['samples']
            )
            euler_angles = _select_equally_spaced_sample(
                combine_data_from(decorations['euler_angles_files'][i]),
                decorations['samples']
            )
            add_body_frame(positions, euler_angles, axis)

    if callbacks is not None:
        for callback in callbacks:
            callback(axis)

    # WAY POINT TRACKER
    axis.text(-1, 0.5, 0, '$P_0$', color='black', fontsize=20)
    axis.text(-1.25, 1, 1.25, '$P_1$', color='black', fontsize=20)
    axis.text(-2.25, 0, 1.75, '$P_2$', color='black', fontsize=20)
    axis.text(-2.25, -2, 2.75, '$P_3$', color='black', fontsize=20)
    axis.text(-1.25, -3, 1.25, '$P_4$', color='black', fontsize=20)
    axis.text(-3.05, -3.8, 2.25, '$P_5$', color='black', fontsize=20)
    add_cylinder(axis, center=(-1, 1, 1), alpha=0.09)
    add_cylinder(axis, center=(-2, 0, 1.5), color='red', alpha=0.09)
    add_cylinder(axis, center=(-2, -2, 2.5), color='green', alpha=0.09)
    add_cylinder(axis, center=(-1, -3, 1), color='orange', alpha=0.09),
    add_cylinder(axis, center=(-2.8, -3.5, 2), color='gray', alpha=0.09)
    add_inertial_frame((-5, -6, 0), axis)

    # TRAJECTORY TRACKING
    # add_inertial_frame((-2, -2, 0), axis)

    # REWARD FUNCTION DIAGRAM
    # add_inertial_frame((-2, -2, 0), axis)
    # add_cylinder(axis, height=1, center=(0, 0, 0.5), alpha=0.1)
    # add_cylinder(axis, radius=2.1, height=1.1, center=(0, 0, 0.55), color='gray', alpha=0.1)
    # axis.scatter(-0.6165, -1.9004, 0.24961, color='black', s=100, marker='o')
    # axis.scatter(1.5998, 1.1824, 0.0093, color='black', s=100, marker='o')
    # axis.scatter(2.25, 0.75, 0.9, color='black', s=150, marker='x')
    # axis.scatter(0.5982, 1.8946, 0.0017, color='black', s=100, marker='o')
    # axis.scatter(0.25, 0.75, 1.5, color='black', s=150, marker='x')
    # add_sphere(axis)
    # add_double_arrow_annotation(
    #     axis,
    #     start=(-0.6489, -1.9971, 1),
    #     end=(-0.618, -1.902, 1),
    #     label='$\delta_R$',
    #     color='red'
    # )
    # add_double_arrow_annotation(
    #     axis,
    #     start=(-2, 0, 1),
    #     end=(-2, 0, 1.1),
    #     label='$\delta_H$',
    #     color='red'
    # )
    # add_double_arrow_annotation(
    #     axis,
    #     start=(-2, 0, 1.1),
    #     end=(-2.1, 0, 1.1),
    #     label='',
    #     color='gray'
    # )
    # add_double_arrow_annotation(
    #     axis,
    #     start=(0.185, -0.65, 0.27),
    #     end=(0, 0, 1),
    #     label='$T_e$',
    #     label_relative_position='left'
    # )
    # add_double_arrow_annotation(
    #     axis,
    #     start=(0, 0, 1),
    #     end=(0.058, 0.080, 1.030),
    #     label='$\Delta_p$', label_position='end',
    # )

    # SUB-TASK 1
    # add_inertial_frame((-0.6, -0.6, 0), axis)
    # add_cylinder(axis, radius=0.025, height=1, center=(0, 0, 0.5))

    # SUB-TASK 2
    # add_inertial_frame((-2, -1.85, 0), axis)
    # add_cylinder(axis, radius=2, height=2, center=(0, 0, 1))

    # SUB-TASK3
    # add_inertial_frame((-1.85, -1.85, 0), axis)
    # add_cylinder(axis, radius=2, height=2, center=(0, 0, 1))

    # SUB-TASKS COMPARISON
    # add_inertial_frame((-1, -0.5, 0), axis)
    # axis.scatter(1, 2, 0.5, color='black', s=100, marker='o')
    # axis.scatter(0, 0, 1, color='black', s=150, marker='x')

    _set_axis(axis, settings)

    plt.show()


def multiple_axis_2d(
        subplots: dict,
        content_specification: dict,
        colors: dict,
        callbacks: list = None
) -> None:
    plt.rcParams['text.usetex'] = True
    fig, axis = plt.subplots(subplots['rows'], subplots['columns'])
    fig.align_labels()

    for col in range(subplots['columns']):
        for row in range(subplots['rows']):
            traces_key = str(f"({row}, {col})")
            if callbacks is not None:
                for callback in callbacks:
                    callback(axis)
            # _add_vertical_lines(axis[row], x_positions=[8.75, 14.87, 23.12, 31.13], label='Disturbances' if row == 0 else '')
            _traces_from_csv(
                content_specification[traces_key]['files'],
                content_specification[traces_key]['labels'],
                axis[row] if subplots['columns'] == 1 else axis[row, col],
                content_specification[traces_key]['references'],
                **colors
            )
            _set_axis(
                axis[row] if subplots['columns'] == 1 else axis[row, col],
                content_specification[traces_key]['settings']
            )

    plt.show()


def animate(data: dict, references: dict, settings: dict, colors: dict, video_name: str = 'video') -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axis = plt.gca()
    figure.subplots_adjust(left=0.13, right=0.87, top=0.85, bottom=0.15)
    _set_axis(axis, **settings)
    parsed_references = _parse_references(references)

    if parsed_references['show']:
        _plot_references(
            parsed_references['files'],
            axis,
            labeled=parsed_references['labeled'],
            label=parsed_references['label']
        )

    def update(frame_number):
        trace.set_xdata(x[:frame_number])
        trace.set_ydata(y[:frame_number])
        return trace

    for i, file in enumerate(data['files']):
        x, y = _get_data_from_csv(file)
        trace = axis.plot(x[0], y[0], colors['color_list'][i] if colors['color_mode'] != 'auto' else '')[0]
        trace.set_label(data['labels'][i])
        axis.legend()
        anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
        anim.save(video_name + str(i) + '.mp4', 'ffmpeg', fps=30, dpi=300)


def animation_3d(
        data: dict,
        references: dict,
        settings: dict,
        color: str = 'red',
        video_name: str = 'video',
        callbacks: list = None,
) -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axis = figure.add_subplot(111, projection='3d')
    axis.view_init(elev=25, azim=40, roll=0)
    _set_axis(axis, settings)

    x, y, z = _get_data_from_csv(data['files'][0])
    add_inertial_frame((1.5, -2, 0), axis)

    # axis.text(-1, 0.65, 0, '$P_0$', color='black', fontsize=18)
    # axis.text(-1, 1.2, 1, '$P_1$', color='black', fontsize=18)
    # axis.text(-2, 0, 1.6, '$P_2$', color='black', fontsize=18)
    # axis.text(-2, -2, 2.6, '$P_3$', color='black', fontsize=18)
    # axis.text(-1, -3.35, 1, '$P_4$', color='black', fontsize=18)
    # axis.text(-2.8, -3.8, 2, '$P_5$', color='black', fontsize=18)
    # add_cylinder(axis, center=(-1, 1, 1))
    # add_cylinder(axis, center=(-2, 0, 1.5), color='red')
    # add_cylinder(axis, center=(-2, -2, 2.5), color='green')
    # add_cylinder(axis, center=(-1, -3, 1), color='orange'),
    # add_cylinder(axis, center=(-2.8, -3.8, 2), color='gray')
    # add_inertial_frame((-5, -6, 0), axis)
    add_sphere(axis, (0.01, -0.01, 1), 0.1)

    if references['show']:
        _traces_from_csv(
            [],
            [''],
            axis,
            references,
            **dict(color_mode='custom', color_list=['black'])
        )

    trace, = axis.plot3D([], [], [], color)
    trace.set_label('Actual Trajectory')

    all_quivers = []

    def update(frame_number):
        nonlocal all_quivers
        trace.set_data(x[:frame_number], y[:frame_number])
        trace.set_3d_properties(z[:frame_number])

        if all_quivers:
            for quiver in all_quivers:
                quiver.remove()

        all_quivers = []
        if callbacks is not None:
            for callback in callbacks:
                quivers = callback(frame_number, axis=axis)
                if quivers:
                    all_quivers.extend(quivers)

        return [trace] + all_quivers

    anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
    anim.save(video_name + '.mp4', 'ffmpeg', fps=30, dpi=300)


def euler_to_rotation_matrix(euler_angles: tuple) -> np.ndarray:
    angles_in_radians= tuple(np.deg2rad(euler_angles))
    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(angles_in_radians[0]), -np.sin(angles_in_radians[0])],
        [0, np.sin(angles_in_radians[0]), np.cos(angles_in_radians[0])]
    ])

    rotation_y = np.array([
        [np.cos(angles_in_radians[1]), 0, np.sin(angles_in_radians[1])],
        [0, 1, 0],
        [-np.sin(angles_in_radians[1]), 0, np.cos(angles_in_radians[1])]
    ])

    rotation_z = np.array([
        [np.cos(angles_in_radians[2]), -np.sin(angles_in_radians[2]), 0],
        [np.sin(angles_in_radians[2]), np.cos(angles_in_radians[2]), 0],
        [0, 0, 1]
    ])

    rotation_matrix = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
    return rotation_matrix


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


def add_double_arrow_annotation(
        axes: plt.Axes,
        start: tuple,
        end: tuple,
        label: str = '',
        label_position: str = 'mid',
        label_relative_position: str = 'none',
        color: str = 'black'
) -> None:
    relative_position = dict(
        none=(0.0, 0.0, 0.0),
        left=(-0.15, -0.1, 0),
        right=(0.15, 0, 0),
        bottom=(0, -0.15, 0),
        top=(0, 0, 0.15),
    )
    axes.quiver(
        start[0], start[1], start[2],
        end[0] - start[0], end[1] - start[1], end[2] - start[2],
        color=color, arrow_length_ratio=0.1, linewidth=1
    )

    axes.quiver(
        end[0], end[1], end[2],
        start[0] - end[0], start[1] - end[1], start[2] - end[2],
        color=color, arrow_length_ratio=0.1, linewidth=1.25
    )

    if label_position == 'mid':
        mid_point = (
            (start[0] + end[0]) / 2 + relative_position[label_relative_position][0],
            (start[1] + end[1]) / 2 + relative_position[label_relative_position][1],
            (start[2] + end[2]) / 2 + relative_position[label_relative_position][2]
        )
        axes.text(mid_point[0], mid_point[1], mid_point[2], label, color='black', fontsize=20)
    elif label_position == 'start':
        axes.text(start[0], start[1], start[2], label, color='black', fontsize=20)
    elif label_position == 'end':
        axes.text(end[0], end[1], end[2], label, color='black', fontsize=20)


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


def compose_sources(parent_directory: str, common_name: str) -> list:
    sources = []

    if not os.path.isdir(parent_directory):
        assert False, "Parent directory doesn't exist"

    for source in os.listdir(parent_directory):
        if common_name in source:
            sources.append(source + '/')

    sources.sort()

    return sources


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def read_from_csv(files: list):
    all_values = []
    steps = None

    for file in files:
        data = pd.read_csv(file)
        if steps is None:
            steps = data['Step']
        all_values.append(data['Value'])

    return steps, pd.DataFrame(all_values)


def calculate_statistics(rewards: pd.DataFrame):
    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)

    return mean_rewards, std_rewards
