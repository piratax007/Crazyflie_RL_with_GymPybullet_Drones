import numpy as np
import os
from datetime import datetime


def create_directory(path: str, comment: str) -> os.path:
    new_directory = os.path.join(path, "save-flight-" + comment + "-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))

    if not os.path.exists(new_directory):
        os.makedirs(new_directory + '/')

    return new_directory


def save_to_csv(points: tuple, path: str, comment: str, file_name: str):
    csv_directory = create_directory(path, comment)

    with open(csv_directory + "/" + file_name + ".csv", 'wb') as out_file:
        np.savetxt(out_file, np.transpose(points), delimiter=",")
