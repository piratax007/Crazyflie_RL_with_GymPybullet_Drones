import numpy as np
import os

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


def compose_sources(parent_directory: str, common_name: str) -> list:
    sources = []

    if not os.path.isdir(parent_directory):
        assert False, "Parent directory doesn't exist"

    for source in os.listdir(parent_directory):
        if common_name in source:
            sources.append(parent_directory + source + '/')
        else:
            print(f'SOMETHING WENT WRONG WITH THE COMMON NAME {source}/')

    sources.sort()

    return sources


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def read_from_csv(files: list):
    all_values = []
    steps = None

    for file in files:
        data = pd.load_csv_from(file)
        if steps is None:
            steps = data['Step']
        all_values.append(data['Value'])

    return steps, pd.DataFrame(all_values)


def calculate_statistics(rewards: pd.DataFrame):
    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)

    return mean_rewards, std_rewards


def get_starting_points(sources: list, csv_filename: str) -> list:
    points = []

    for source in sources:
        csv_path = os.path.join(source, csv_filename)

        if os.path.isfile(csv_path):
            with open(csv_path, newline='') as csv_file:
                reader = csv.reader(csv_file)
                first_raw = next(reader, None)
                if first_raw:
                    points.append(tuple(float(coordinate) for coordinate in first_raw))

    return points