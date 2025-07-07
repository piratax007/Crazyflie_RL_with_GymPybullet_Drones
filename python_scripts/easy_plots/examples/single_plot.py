#!/usr/bin/env python3

import os
from aquarel import load_theme
from python_scripts.easy_plots.easy_plots import (
    GENERAL_PATH,
    single_axes_2d
)

if __name__ == "__main__":
    GENERAL_PATH.path = os.path.dirname(
        'python_scripts/easy_plots/examples/data/'
    )
    theme = (
        load_theme('scientific')
        .set_font(family='serif', size=22)
        .set_axes(bottom=True, top=True, left=True, right=True, xmargin=0, ymargin=0, zmargin=0, width=2)
        .set_grid(style='--', width=1)
        .set_ticks(draw_minor=True, pad_major=10)
        .set_lines(width=2.5)
        .set_legend(location='upper right', alpha=0)
    )
    theme.apply()

    single_axes_2d(
        files = ['x0.csv', 'y0.csv', 'z0.csv'],
        labels = ['x', 'y', 'z'],
        references = dict(
            show = False,
            interior_detail = False,
        ),
        colors = dict(
            color_mode = 'auto'
        ),
        settings = dict(
            limits = dict(
                mode = 'auto',
            ),
            labels = dict(
                x_label = 'time (s)',
                y_label = 'position (m)',
                title = 'Single Plot Example'
            )
        )
    )