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
        axes: plt.Axes,
        labeled: bool = True,
        label: str = 'Reference',
        style: str = '--'
) -> None:
    for i, file in enumerate(files):
        reference = _get_data_from_csv(file)
        if style == "point" and len(reference) == 3:
            axes.scatter(reference[0], reference[1], reference[2], color='black', label=label, s=100)
        else:
            axes.plot(
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

# TODO: [A] To be improved the positioning and setup of the legend
def _get_legend_settings(mode: str) -> dict:
    legend_settings = {
        'NONE': {},
        '2D': {
            'bbox_to_anchor': (0, 0.85, 1, 0.75),
            'loc': "lower left",
            'borderaxespad': 0,
            'ncol': 4
        },
        '3D': {
            'bbox_to_anchor': (0, 1, 1, 0.25),
            'loc': "lower right",
            'borderaxespad': 0.5,
            'ncol': 4
        }
    }

    return legend_settings[mode]

def _traces_from_csv(
        files: list,
        labels: list,
        axes: plt.Axes,
        references: dict,
        line_style: str = '-',
        legend_mode: str = '2D',
        **colors: dict,
) -> None:
    parsed_references = _parse_references(references)

    interior_axes = _interior_axes(
        parsed_references['interior_detail'],
        axes,
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
            axes.indicate_inset_zoom(interior_axes, edgecolor='lightgray', alpha=0.25)
        axes.plot(
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
            axes,
            parsed_references['labeled'],
            parsed_references['label'],
            parsed_references['style']
        )

    # axes.legend()

    axes.legend(**_get_legend_settings(legend_mode))


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


def _parse_settings(settings: dict) -> dict:
    if 'axes_format' not in settings:
        settings['axes_format'] = 'plain'

    if 'axes_aspect' not in settings:
        settings['axes_aspect'] = 'auto'

    return settings


def _set_axes(axes: plt.Axes, settings: dict) -> None:
    # ToDo: Improve exceptions
    parsed_settings = _parse_settings(settings)

    if parsed_settings['limits']['mode'] != 'auto':
        axes.set_xlim(parsed_settings['limits']['x_range'])
        axes.set_ylim(parsed_settings['limits']['y_range'])
        try:
            axes.set_zlim(parsed_settings['limits']['z_range'])
        except:
            pass

    axes.set_xlabel(parsed_settings['labels']['x_label'], labelpad=15)
    axes.set_ylabel(parsed_settings['labels']['y_label'], labelpad=15)
    try:
        axes.set_zlabel(parsed_settings['labels']['z_label'], labelpad=20)
    except:
        pass

    axes.set_title(parsed_settings['labels']['title'])
    axes.ticklabel_format(axis='both', style=parsed_settings['axes_format'], scilimits=(0, 0))
    axes.set_aspect(parsed_settings['axes_aspect'])
    # axes.set_box_aspect([6,9,4])


def single_axes_2d(files: list, labels: list, references: dict, colors: dict, settings: dict, callbacks: list = None) -> None:
    _, axes = plt.subplots(1)

    _traces_from_csv(files, labels, axes, references, legend_mode='NONE', **colors)
    _set_axes(axes, settings)

    if callbacks is not None:
        for callback in callbacks:
            callback(axes)

    plt.show()


def _select_equally_spaced_sample(data: tuple, sample_size: int) -> tuple:
    step = len(data[0]) // sample_size
    r = tuple(tuple(inner_tuple[i * step] for i in range(sample_size)) for inner_tuple in data)
    return r


def single_axes_3d(
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
    axes = figure.add_subplot(projection='3d')

    _traces_from_csv(
        files,
        labels,
        axes,
        references,
        line_style=settings['line_style'] if 'line_style' in settings else '-',
        line_width=settings['line_width'] if 'line_width' in settings else 1.5,
        legend_mode='3D',
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

    if callbacks is not None:
        for callback in callbacks:
            callback(axes)

    # TODO: Move the following markers as examples in the documentation
    # axes.scatter(-0.6165, -1.9004, 0.24961, color='black', s=100, marker='o')
    # axes.scatter(1.5998, 1.1824, 0.0093, color='black', s=100, marker='o')
    # axes.scatter(2.25, 0.75, 0.9, color='black', s=150, marker='x')
    # axes.scatter(0.5982, 1.8946, 0.0017, color='black', s=100, marker='o')
    # axes.scatter(0.25, 0.75, 1.5, color='black', s=150, marker='x')

    # axes.scatter(1, 2, 0.5, color='black', s=100, marker='o')
    # axes.scatter(0, 0, 1, color='black', s=150, marker='x')

    _set_axes(axes, settings)

    plt.show()


def _axes_handler(axes, axes_arrangement: dict, row_index: int, column_index: int) -> plt.Axes:
    if axes_arrangement['columns'] == 1:
        return axes[row_index]

    if axes_arrangement['rows'] == 1:
        return axes[column_index]

    return axes[row_index, column_index]


def multiple_axes_2d(
        subplots: dict,
        content_specification: dict,
        colors: dict,
        callbacks: list = None
) -> None:
    plt.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(subplots['rows'], subplots['columns'])
    fig.align_labels()

    for col in range(subplots['columns']):
        for row in range(subplots['rows']):
            traces_key = str(f"({row}, {col})")
            if callbacks is not None:
                for callback in callbacks:
                    callback(axes)

            _traces_from_csv(
                content_specification[traces_key]['files'],
                content_specification[traces_key]['labels'],
                _axes_handler(axes, subplots, row, col),
                content_specification[traces_key]['references'],
                **colors
            )
            _set_axes(
                _axes_handler(axes, subplots, row, col),
                content_specification[traces_key]['settings']
            )

    plt.show()


def animate(data: dict, references: dict, settings: dict, colors: dict, video_name: str = 'video') -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axes = plt.gca()
    figure.subplots_adjust(left=0.13, right=0.87, top=0.85, bottom=0.15)
    _set_axes(axes, settings)
    parsed_references = _parse_references(references)

    if parsed_references['show']:
        _plot_references(
            parsed_references['files'],
            axes,
            labeled=parsed_references['labeled'],
            label=parsed_references['label']
        )

    def update(frame_number):
        trace.set_xdata(x[:frame_number])
        trace.set_ydata(y[:frame_number])
        return trace

    for i, file in enumerate(data['files']):
        x, y = _get_data_from_csv(file)
        trace = axes.plot(x[0], y[0], colors['color_list'][i] if colors['color_mode'] != 'auto' else '')[0]
        trace.set_label(data['labels'][i])
        axes.legend()
        anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
        anim.save(video_name + str(i) + '.mp4', 'ffmpeg', fps=30, dpi=300)

# TODO: add callbacks argument
def animation_3d(
        data: dict,
        references: dict,
        settings: dict,
        color: str = 'red',
        video_name: str = 'video',
        callbacks: list = None,
) -> None:
    figure = plt.figure(figsize=(16, 9), dpi=720 / 16)
    axes = figure.add_subplot(111, projection='3d')
    axes.view_init(elev=25, azim=40, roll=0)
    _set_axes(axes, settings)

    x, y, z = _get_data_from_csv(data['files'][0])

    if references['show']:
        _traces_from_csv(
            [],
            [''],
            axes,
            references,
            **dict(color_mode='custom', color_list=['black'])
        )

    trace, = axes.plot3D([], [], [], color)
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
                quivers = callback(frame_number, axes=axes)
                if quivers:
                    all_quivers.extend(quivers)

        return [trace] + all_quivers

    anim = animation.FuncAnimation(figure, update, frames=len(x), interval=3, repeat=False)
    anim.save(video_name + '.mp4', 'ffmpeg', fps=60, dpi=300)
