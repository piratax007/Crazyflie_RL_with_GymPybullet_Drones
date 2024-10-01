#!/usr/bin/env python3

import os
from aquarel import load_theme
from easy_plots import multiple_axis_2D, GENERAL_PATH

if __name__ == '__main__':
    GENERAL_PATH.path = os.path.dirname(
        '../../experiments/results/save-Test-NEW-FROM-SCRATCH-03.27.2024_09.34.31/'
    )

    theme = (
        load_theme('scientific')
        .set_font(family='serif', size=20)
        .set_title(pad=20)
        .set_axes(bottom=True, top=True, left=True, right=True, xmargin=0, ymargin=0, zmargin=0, width=2)
        .set_grid(style='--', width=1)
        .set_ticks(draw_minor=True, pad_major=10)
        .set_lines(width=2.5)
    )
    theme.apply()

    # animate(files=[
    #     'save-flight-starting-from-x0y1z0-03.19.2024_15.26.29/x0.csv',
    #     'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x0.csv'
    # ],
    #     references=dict(
    #         view=True,
    #         files=[
    #             'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x_low_ref.csv',
    #             'save-flight-starting-from-x05ym05z1.5-03.27.2024_10.14.42/x_high_ref.csv',
    #         ]
    #     ),
    #     settings=dict(
    #         limits=dict(
    #             mode='custom',
    #             x_range=(0, 15),
    #             y_range=(-1, 1)
    #         ),
    #         labels=dict(
    #             x_label='t (s)',
    #             y_label='y (m)',
    #             title="Flight starting from random x position"
    #         )
    #     ),
    #     colors=dict(
    #         color_mode='custom',
    #         color_list=['red', 'green']
    #     )
    # )

    multiple_axis_2D(
        subplots=dict(rows=3, columns=1),
        content_specification={
            '(0, 0)': dict(
                files=[
                    'save-flight-from_scratch-03.27.2024_16.48.35/r0.csv'
                ],
                references=dict(
                    view=True,
                    files=[
                        'save-flight-from_scratch-03.27.2024_16.48.35/x_low_ref.csv',
                        'save-flight-from_scratch-03.27.2024_16.48.35/x_high_ref.csv',
                    ]
                ),
                settings=dict(
                    limits=dict(
                        mode='custom',
                        x_range=(0, 15),
                        y_range=(-2, 2)
                    ),
                    labels=dict(
                        x_label='t (s)',
                        y_label='roll (deg)',
                        title=''
                    )
                )),
            '(1, 0)': dict(
                files=[
                    'save-flight-from_scratch-03.27.2024_16.48.35/p0.csv'
                ],
                references=dict(
                    view=True,
                    files=[
                        'save-flight-from_scratch-03.27.2024_16.48.35/y_low_ref.csv',
                        'save-flight-from_scratch-03.27.2024_16.48.35/y_high_ref.csv',
                    ]
                ),
                settings=dict(
                    limits=dict(
                        mode='custom',
                        x_range=(0, 15),
                        y_range=(-2, 2)
                    ),
                    labels=dict(
                        x_label='t (s)',
                        y_label='pitch (deg)',
                        title=''
                    )
                )),
            '(2, 0)': dict(
                files=[
                    'save-flight-from_scratch-03.27.2024_16.48.35/ya0.csv',
                ],
                references=dict(
                    view=True,
                    files=[
                        'save-flight-from_scratch-03.27.2024_16.48.35/z_low_ref.csv',
                        'save-flight-from_scratch-03.27.2024_16.48.35/z_high_ref.csv',
                    ]
                ),
                settings=dict(
                    limits=dict(
                        mode='custom',
                        x_range=(0, 15),
                        y_range=(0, 100)
                    ),
                    labels=dict(
                        x_label='t (s)',
                        y_label='yaw (deg)',
                        title=''
                    )
                )),
        },
        colors=dict(color_mode='custom', color_list=['red']),
    )
