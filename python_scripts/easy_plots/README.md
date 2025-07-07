# HOW TO USE `easy_plots.py`

`easy_plots.py` can be used to plot time series from CSV files with one or multiple traces using the same figure or a 
set of subfigures, in both 2D and 3D. `easy_plots` require [`aquarel`][2] to use beautiful themes and as a way to easily 
define new themes, whether you don't want to install that additional package, you should use the `matplotlibrc` file 
to use a predefine theme compatible with IEEE publications or to define one by yourself.

## How to use `easy_plots`
Prepare a new python script. Use the `GENERAL_PATH` object imported from easy plots to set the path to the root 
directory of the CSV files, this is use to easily point to the CSV files without re-write the full path.

```Python
import os
from easy_plots import GENERAL_PATH
...
GENERAL_PATH.path = os.path.dirname(
        'path/to/the/CSV/files/'
    )
```

Using `easy_plots` you can plot one or multiple time series in the same figure or 
in a set of sub-figures. It can be used to plot 2D and 3D traces, and to record animations. Can be used callback 
functions that receive an `plt.axes` object as argument, to add specific details to the plot e.g. coordinate body 
frames.

With `aquarel` you can use a set of beautiful themes, create your own theme or modify existing ones.
If you prefer to avoid the installation of the aquarel package, you can use the `matplotlibrc` to manage the 
appearance of the plots. Easy plots includes a `matplotlibrc` file with a theme compatible with IEEE 
publications.

## EXAMPLES

|   | Description                                                                   |
|---|-------------------------------------------------------------------------------|
|   | One figure with three traces using automatic axis length and color selection. |
|   |                                                                               |
|   |                                                                               |

**NOTE:** The `matplotlibrc` file

[1]: https://matplotlib.org/stable/gallery/color/named_colors.html
[2]: https://github.com/lgienapp/aquarel?tab=readme-ov-file