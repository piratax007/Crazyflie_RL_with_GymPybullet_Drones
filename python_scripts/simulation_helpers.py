import numpy as np
import os
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R
import pybullet as p
from collections import deque
from typing import Deque, Callable
from numpy.typing import NDArray

def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(model_class, policy_path, model):
    if os.path.isfile(policy_path + '/' + model):
        return model_class.load(policy_path + '/' + model)

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def helix_trajectory(number_of_points: int = 50, radius: int = 2, angle_range: float = 30.0) -> tuple:
    """
    Generates a set of 3D helix trajectory coordinates and target yaw angles within a specified angle range.
    The helicoidal trajectory is defined by a circular motion in the XY-plane with linear progression in the
    Z direction. The target yaw angles are intended to be follow over the trajectory.

    Args:
        number_of_points (int): Number of points in the trajectory.
        radius (int): Radius of the helix, defining the circular XY-plane motion.
        angle_range (float): Maximum angle range in degrees for yaw oscillation.

    Returns:
        tuple: A tuple containing the following:
            - x_coordinates (numpy.ndarray): x-coordinates of the trajectory.
            - y_coordinates (numpy.ndarray): y-coordinates of the trajectory.
            - z_coordinates (numpy.ndarray): z-coordinates of the trajectory.
            - yaw_angles (numpy.ndarray): Yaw angles of the trajectory with oscillation.

    How to:
    1. Define the initial point for the helix trajectory in the initializer of the test_env:
    ```
    ...
    test_env = test_env(
        initial_xyzs=np.array([[-1.0, 0.0, 0.0]]),
        ...
    ```
    2. Before the simulation loop, call the helix_trajectory function:
    ```
    ...
    x_target, y_target, z_target, yaw_target = helix_trajectory(simulation_seconds, 1)

    for i in range(simulation_seconds):
        ...
    ```
    3. At the beginning of the simulation loop, update the observation vector:
    ```
    ...
    for i in range(simulation_seconds):
        obs[0][0] += x_target[i]
        obs[0][1] += y_target[i]
        obs[0][2] -= z_target[i]
        ...
    ```
    """
    angles = np.linspace(0, 4 * np.pi, number_of_points)
    x_coordinates = radius * np.cos(angles)
    y_coordinates = radius * np.sin(angles)
    z_coordinates = np.linspace(0, 1, number_of_points)

    yaw_angles = np.arctan2(y_coordinates, x_coordinates)

    angle_range_rad = np.radians(angle_range)

    oscillation = angle_range_rad * np.sin(np.linspace(0, 4 * np.pi, number_of_points))

    yaw_angles += oscillation

    yaw_angles = np.clip(yaw_angles, -angle_range_rad, angle_range_rad)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles


def lemniscate_trajectory(number_of_points: int = 50, a: float = 2) -> tuple:
    """
    Generates a lemniscate trajectory with specified number of points and scale.
    The function computes the x, y, and z coordinates along a lemniscate curve,
    as well as the yaw angles at each point of the trajectory. The lemniscate
    is scaled by the parameter `a`.

    Args:
        number_of_points: int
            The number of points to compute along the trajectory. Defaults to 50.
        a: float
            The scale of the lemniscate. Defines the size of the curve. Defaults to 2.

    Returns:
        tuple:
            A tuple containing four elements:
            - x_coordinates (numpy.ndarray): The x-coordinates of the trajectory points.
            - y_coordinates (numpy.ndarray): The y-coordinates of the trajectory points.
            - z_coordinates (numpy.ndarray): The z-coordinates (zeros) for the trajectory points.
            - yaw_angles (numpy.ndarray): The yaw angles for the trajectory points.

    How to:
    How to:
    1. Define the initial point for the helix trajectory in the initializer of the test_env:
    ```
    ...
    test_env = test_env(
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
        ...
    ```
    2. Before the simulation loop, call the lemniscate_trajectory function:
    ```
    ...
    x_target, y_target, z_target, yaw_target = lemniscate_trajectory(simulation_seconds, 1)

    for i in range(simulation_seconds):
        ...
    ```
    3. At the beginning of the simulation loop, update the observation vector:
    ```
    ...
    for i in range(simulation_seconds):
        obs[0][0] += x_target[i]
        obs[0][1] += y_target[i]
        obs[0][2] -= z_target[i]
        ...
    ```
    """
    t = np.linspace(0, 2.125 * np.pi, number_of_points)
    x_coordinates = a * np.sin(t) / (1 + np.cos(t) ** 2)
    y_coordinates = a * np.sin(t) * np.cos(t) / (1 + np.cos(t) ** 2)
    z_coordinates = np.zeros(number_of_points)

    yaw_angles = np.arctan2(-y_coordinates, -x_coordinates)

    return x_coordinates, y_coordinates, z_coordinates, yaw_angles


def smooth_trajectory(points: list, num_points: int = 100, target_center: tuple = None,
                      yaw_mode: str = "tangent"):
    """
    Generates a smooth trajectory through a series of 3D points and calculates the corresponding roll,
    pitch, and yaw angles for each point.

    The function takes a list of 3D points, interpolates a smooth trajectory through the points
    using a spline, and computes the yaw angles at each interpolated point based on the specified mode.

    Arguments:
        points (list): A list of 3D points represented as [x, y, z] coordinates through which the
                       trajectory is interpolated.
        num_points (int): The number of interpolated points generated along the trajectory. Defaults
                          to 100.
        target_center (tuple, optional): A 3D point [x, y, z] used as reference for yaw calculations.
                                        Required when yaw_mode is "towards_center" or "radial".
                                        Defaults to None.
        yaw_mode (str): Mode for calculating yaw angles:
                       - "tangent": Yaw follows the trajectory tangent (default)
                       - "towards_center": Yaw points towards target_center
                       - "radial": Yaw matches the radial angle from target_center (for circular arcs)

    Returns:
        tuple: A tuple containing four elements:
               - A tuple of x-coordinates along the smoothed trajectory.
               - A tuple of y-coordinates along the smoothed trajectory.
               - A tuple of z-coordinates along the smoothed trajectory.
               - A tuple of yaw angles associated with each trajectory point.

    How To:
    1. Before the simulation loop define a list of points (at least four)
    ```
    ...
    points = [
        [0, 0, 0],
        [1, 2, 0.5],
        [2, 4, 1],
        [3, 2, 1.5],
        [4, 0, 2],
        [5, -2, 2.5],
        [6, -4, 3],
        [7, -2, 3.5],
        [8, 0, 4]
    ]

    for i in range(simulation_seconds):
        ...
    ```
    2. After the list of points, and before the simulation loop, call the smooth_trajectory function:
    ```
    ...
    # For circular inspection (yaw matches angular position):
    x_target, y_target, z_target, yaw_target = smooth_trajectory(
        points, simulation_seconds, target_center=(0, 0, 1), yaw_mode="radial")

    # For inspection task (yaw points towards center):
    x_target, y_target, z_target, yaw_target = smooth_trajectory(
        points, simulation_seconds, target_center=(0, 0, 1), yaw_mode="towards_center")

    # For regular trajectory (yaw follows path):
    x_target, y_target, z_target, yaw_target = smooth_trajectory(points, simulation_seconds)

    for i in range(simulation_seconds):
        ...
    ```
    3. At the beginning of the simulation loop, update the observation vector:
    ```
    ...
    for i in range(simulation_seconds):
        obs[0][0] += x_target[i]
        obs[0][1] += yaw_target[i]
        obs[0][2] -= z_target[i]
        ...
    ```
    """
    points = np.array(points)
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], s=10)
    u_fine = np.linspace(0, 1, num_points)
    x_fine, y_fine, z_fine = splev(u_fine, tck)
    smooth_points = np.vstack((x_fine, y_fine, z_fine)).T

    yaw_angles = []

    if yaw_mode == "towards_center" and target_center is not None:
        # Calculate yaw angles pointing towards the target center
        cx, cy, cz = target_center
        for point in smooth_points:
            # Vector from current point to center
            dx = cx - point[0]
            dy = cy - point[1]
            # Calculate yaw angle (atan2 gives angle in x-y plane)
            yaw = np.arctan2(dy, dx)
            yaw_angles.append(yaw)
    elif yaw_mode == "radial" and target_center is not None:
        # Calculate yaw angles matching the radial angle from center
        cx, cy, cz = target_center
        for point in smooth_points:
            # Vector from center to current point
            dx = point[0] - cx
            dy = point[1] - cy
            # Calculate yaw angle (atan2 gives angle in x-y plane)
            yaw = np.arctan2(dy, dx)
            yaw_angles.append(yaw)
    else:
        # Original behavior: yaw follows trajectory tangent
        tangents = np.diff(smooth_points, axis=0)
        tangents /= np.linalg.norm(tangents, axis=1)[:, None]
        tangents = np.vstack((tangents, tangents[-1]))

        roll_pitch_yaw = []

        for i in range(len(smooth_points)):
            t = tangents[i]
            y_axis = t
            z_axis = np.array([0, 0, 1])
            if np.allclose(y_axis, z_axis):
                z_axis = np.array([0, 1, 0])
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)

            rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
            r = R.from_matrix(rotation_matrix)
            roll, pitch, yaw = r.as_euler('xyz', degrees=False)

            roll_pitch_yaw.append((roll, pitch, yaw))

        yaw_angles = [yaw for _, _, yaw in roll_pitch_yaw]

    x_tuple = tuple(smooth_points[:, 0])
    y_tuple = tuple(smooth_points[:, 1])
    z_tuple = tuple(smooth_points[:, 2])
    yaw_tuple = tuple(yaw_angles)

    return x_tuple, y_tuple, z_tuple, yaw_tuple


def random_cylindrical_positions(
        inner_radius: float = 0.0,
        outer_radius: float = 1.5,
        cylinder_height: float = 1.5,
        cylinder_center: tuple = (0, 0, 1),
        mode: str = "inside",
        min_distance: float = 0.0,
        max_distance: float = 0.0
) -> tuple:
    """
    Generates random cylindrical positions based on specified constraints.

    This function generates random positions within or outside a cylindrical region.
    The cylinder is defined by its geometry and location in 3D space. Additional
    parameters control the distance and mode of generation. Positions are returned
    as a tuple of x, y, and z coordinates.

    Parameters:
        inner_radius (float): Defines the minimum radius from the cylinder's central
            axis for generating points. Default is 0.0.
        outer_radius (float): Defines the maximum radius from the cylinder's central
            axis for generating points. Default is 1.5.
        cylinder_height (float): Height of the cylinder, defining the range along
            the z-axis. Default is 1.5.
        cylinder_center (tuple): A tuple (cx, cy, cz) representing the center of the
            cylinder. Default is (0, 0, 1).
        mode (str): Specifies the mode of position generation. Can be either "inside"
            to generate points within the cylinder or "outside" for points outside
            the defined cylindrical shell. Default is "inside".
        min_distance (float): The minimum distance outside the cylinder's outer surface
            for the "outside" mode. Default is 0.0.
        max_distance (float): The maximum distance outside the cylinder's outer surface
            for the "outside" mode. Default is 0.0.

    Returns:
        tuple: A tuple containing x, y, and z coordinates of the generated random
        position in 3D space.

    How to:
        Initialize the position of the drone in the initializer of the test_env:
        ```
        ...
        test_env = test_env(
            initial_xyzs = np.array([[*random_cylindrical_positions(outer_radius=2.0, cylinder_height=2, mode='inside')]]),
            ...
        ```
    """
    cx, cy, cz = cylinder_center

    if mode == "inside":
        r = np.sqrt(np.random.uniform(inner_radius ** 2, outer_radius ** 2))
    elif mode == "outside":
        r = np.sqrt(np.random.uniform((outer_radius + min_distance) ** 2, (outer_radius + max_distance) ** 2))
    else:
        r = 0

    theta = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(-cylinder_height / 2, cylinder_height / 2 + max_distance)

    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = cz + z

    return x, y, z


def plot_dashed_line(points, dash_length=0.1, gap_length=0.05, color=(0.5, 0.5, 0.5), line_width=2.0):
    """
    Plots a dashed gray line connecting a list of 3D points.

    This function creates a visual dashed line by drawing multiple small line segments
    with gaps between them, connecting the provided sequence of points.

    Parameters:
        points (list): A list of 3D positions [x, y, z] to connect with dashed lines.
        dash_length (float, optional): Length of each dash segment in meters. Defaults to 0.1.
        gap_length (float, optional): Length of each gap between dashes in meters. Defaults to 0.05.
        color (Sequence[float], optional): RGB color of the dashed line. Defaults to (0.5, 0.5, 0.5) (gray).
        line_width (float, optional): Width of the line segments. Defaults to 1.

    Returns:
        list: List of IDs for all the line segments that form the dashed line.

    Raises:
        ValueError: If fewer than 2 points are provided.

    How to:
        Before the simulation loop, define a list of points and call the function:
        ```
        ...
        waypoints = [
            [0.0, 0.0, 1.0],
            [0.7, 0.0, 1.0],
            [1.3, 0.9, 0.75],
            [2.1, 0.6, 1.5]
        ]

        dashed_line_ids = plot_dashed_line(waypoints, dash_length=0.08, gap_length=0.04)

        for i in range(simulation_seconds):
            ...
        ```
    """
    if len(points) < 2:
        raise ValueError("At least 2 points are required to plot a dashed line")

    points = np.array(points)
    line_ids = []

    # Iterate through consecutive point pairs
    for i in range(len(points) - 1):
        start_point = points[i]
        end_point = points[i + 1]

        # Calculate segment direction and length
        segment_vector = end_point - start_point
        segment_length = np.linalg.norm(segment_vector)
        segment_direction = segment_vector / segment_length

        # Current position along the segment
        current_distance = 0.0

        # Draw dashes along this segment
        while current_distance < segment_length:
            # Calculate dash start and end positions
            dash_start = start_point + current_distance * segment_direction
            dash_end_distance = min(current_distance + dash_length, segment_length)
            dash_end = start_point + dash_end_distance * segment_direction

            # Draw the dash
            line_id = p.addUserDebugLine(
                lineFromXYZ=dash_start,
                lineToXYZ=dash_end,
                lineColorRGB=color,
                lineWidth=line_width,
                lifeTime=0
            )
            line_ids.append(line_id)

            # Move to next dash (skip the gap)
            current_distance += dash_length + gap_length

    return line_ids


def add_way_point(position, radius=0.1, arrow=None):
    """
    Adds a waypoint in the form of a sphere to the simulation. This function
    creates a sphere visual shape with specified parameters and places it at
    the given position. Optionally, an arrow can be added to indicate direction.

    Parameters:
        position (Sequence[float]): A 3D position [x, y, z] where the sphere will
            be placed.
        radius (float, optional): Radius of the sphere to represent the waypoint.
            Defaults to 0.1.
        color (Sequence[float], optional): RGBA color of the sphere. Defaults to
            (1, 0, 0, 0.5).
        arrow (dict, optional): Dictionary with keys "show_arrow" (bool) and "angle" (float).
            If show_arrow is True, draws an arrow with length 2*radius pointing in the
            direction given by angle (in radians) in the x-y plane. Defaults to None.

    Returns:
        int or tuple: The ID of the created sphere, or a tuple (sphere_id, arrow_line_id)
            if an arrow was added.

    Raises:
        Exception: If the simulation environment is not set up properly.

    How to:
        Before the simulation loop, call the add_way_point function once for each waypoint:
        ```
        ...
        _ = {
            0: add_way_point((-1, 1, 0), radius=0.025),
            1: add_way_point((-1, 1, 1), radius=0.025, arrow={"show_arrow": True, "angle": 0.58}),
            2: add_way_point((-2, 0, 1.5), radius=0.025)
        }

        for i in range(simulation_seconds):
            ...
        ```
    """
    sphere_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1.0, 0.43, 0.1, 0.5],
    )
    sphere_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=sphere_visual,
        basePosition=position,
    )

    if arrow and arrow.get("show_arrow", False):
        angle = arrow.get("angle", 0.0)
        arrow_length = 5 * radius
        cone_length = 1.5 * radius
        cone_radius = 0.5 * radius

        # Calculate arrow endpoint in x-y plane
        end_x = position[0] + arrow_length * np.cos(angle)
        end_y = position[1] + arrow_length * np.sin(angle)
        end_z = position[2]

        # Draw arrow as a line
        arrow_line_id = p.addUserDebugLine(
            lineFromXYZ=position,
            lineToXYZ=[end_x, end_y, end_z],
            lineColorRGB=[1.0, 0.43, 0.1],
            lineWidth=2,
            lifeTime=0
        )

        # Draw cone at the arrow tip using 8 lines
        cone_tip = np.array([end_x, end_y, end_z])
        # Calculate cone base center (slightly back from tip)
        cone_base_center = np.array([
            position[0] + (arrow_length - cone_length) * np.cos(angle),
            position[1] + (arrow_length - cone_length) * np.sin(angle),
            position[2]
        ])

        # Create perpendicular vectors for the cone base circle
        # Direction vector of the arrow
        arrow_dir = np.array([np.cos(angle), np.sin(angle), 0])
        # Perpendicular vector 1 (rotated 90 degrees in x-y plane)
        perp1 = np.array([-np.sin(angle), np.cos(angle), 0])
        # Perpendicular vector 2 (up direction)
        perp2 = np.array([0, 0, 1])

        cone_line_ids = []
        # Create 8 lines around the cone
        for i in range(8):
            circle_angle = 2 * np.pi * i / 8
            # Calculate base point on the circle
            base_point = cone_base_center + cone_radius * (
                    np.cos(circle_angle) * perp1 + np.sin(circle_angle) * perp2
            )
            # Draw line from base point to tip
            cone_line_id = p.addUserDebugLine(
                lineFromXYZ=base_point,
                lineToXYZ=cone_tip,
                lineColorRGB=[1.0, 0.43, 0.1],
                lineWidth=2,
                lifeTime=0
            )
            cone_line_ids.append(cone_line_id)

        return sphere_id, arrow_line_id, cone_line_ids

    return sphere_id


def add_cylindrical_obstacle(x, y, radius, height, color=(0.5, 0.5, 0.5, 1.0)):
    """
    Adds a rigid cylindrical obstacle to the simulation environment. The cylinder
    has both visual and collision shapes, allowing drones to physically collide
    with it.

    Parameters:
        x (float): X-coordinate of the cylinder's center position.
        y (float): Y-coordinate of the cylinder's center position.
        radius (float): Radius of the cylinder in meters.
        height (float): Height of the cylinder in meters.
        color (Sequence[float], optional): RGBA color of the cylinder.
            Defaults to (0.5, 0.5, 0.5, 1.0) (gray).

    Returns:
        int: The ID of the created cylinder in the simulation.

    Raises:
        Exception: If PyBullet simulation is not initialized.

    How to:
        Before the simulation loop, call the add_cylindrical_obstacle function
        for each obstacle:
        ```
        ...
        obstacles = [
            add_cylindrical_obstacle(x=1.0, y=0.0, radius=0.2, height=2.0),
            add_cylindrical_obstacle(x=-1.0, y=1.0, radius=0.15, height=1.5),
            add_cylindrical_obstacle(x=0.0, y=-1.0, radius=0.25, height=1.8)
        ]

        for i in range(simulation_seconds):
            ...
        ```
    """
    # Create collision shape for physical interactions
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    # Create visual shape for rendering
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    # Create the multi-body with both collision and visual shapes
    # Position z is set to height/2 so the cylinder sits on the ground (z=0)
    cylinder_id = p.createMultiBody(
        baseMass=0,  # Mass of 0 makes it static/immovable
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[x, y, height / 2]
    )

    return cylinder_id


def add_rectangular_wall(position, width, height, thickness=0.1, orientation_deg=0.0,
                         color=(0.5, 0.5, 0.5, 1.0)):
    """
    Adds a rectangular wall obstacle to the simulation environment. The wall is
    a rigid box with collision and visual properties, allowing physical interactions.

    Parameters:
        position (Sequence[float]): A 3D position [x, y, z] for the center of the wall.
        width (float): Width of the wall in meters (along its length).
        height (float): Height of the wall in meters (vertical dimension).
        thickness (float, optional): Thickness of the wall in meters. Defaults to 0.1.
        orientation_deg (float, optional): Rotation angle around the Z-axis in degrees.
            0 degrees aligns the wall with the X-axis. Defaults to 0.0.
        color (Sequence[float], optional): RGBA color of the wall.
            Defaults to (0.5, 0.5, 0.5, 1.0) (gray).

    Returns:
        int: The ID of the created wall in the simulation.

    Raises:
        Exception: If PyBullet simulation is not initialized.

    How to:
        Before the simulation loop, call the add_rectangular_wall function:
        ```
        ...
        # Create a wall at position (2, 0, 1) with width 3m, height 2m
        wall_id = add_rectangular_wall(
            position=(2, 0, 1),
            width=3.0,
            height=2.0,
            thickness=0.15,
            orientation_deg=45.0,
            color=(0.7, 0.3, 0.3, 1.0)
        )

        for i in range(simulation_seconds):
            ...
        ```
    """
    # Create collision shape for physical interactions
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[width / 2, thickness / 2, height / 2]
    )

    # Create visual shape for rendering
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[width / 2, thickness / 2, height / 2],
        rgbaColor=color
    )

    # Calculate orientation quaternion
    orientation_rad = np.deg2rad(orientation_deg)
    orientation_quat = p.getQuaternionFromEuler([0, 0, orientation_rad])

    # Create the multi-body with both collision and visual shapes
    wall_id = p.createMultiBody(
        baseMass=0,  # Mass of 0 makes it static/immovable
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position,
        baseOrientation=orientation_quat
    )

    return wall_id


def add_circular_wall(center, radius, height, start_angle_deg, end_angle_deg,
                      thickness=0.1, num_segments=20, color=(0.5, 0.5, 0.5, 1.0)):
    """
    Adds a circular wall segment to the simulation environment. The wall follows
    an arc defined by a center point, radius, and angular range.

    Parameters:
        center (Sequence[float]): A 3D position [x, y, z] for the center of the arc.
        radius (float): Radius of the circular arc in meters.
        height (float): Height of the wall in meters.
        start_angle_deg (float): Starting angle of the arc in degrees (0 = +X axis).
        end_angle_deg (float): Ending angle of the arc in degrees.
        thickness (float, optional): Thickness of the wall in meters. Defaults to 0.1.
        num_segments (int, optional): Number of box segments to approximate the arc.
            Defaults to 20.
        color (Sequence[float], optional): RGBA color of the wall.
            Defaults to (0.5, 0.5, 0.5, 1.0) (gray).

    Returns:
        list: List of IDs for all the box segments that form the wall.

    Raises:
        Exception: If PyBullet simulation is not initialized.

    How to:
        Before the simulation loop, call the add_circular_wall function:
        ```
        ...
        wall_ids = add_circular_wall(
            center=(0, 0, 1),
            radius=3.5,
            height=2.0,
            start_angle_deg=-15,
            end_angle_deg=15
        )

        for i in range(simulation_seconds):
            ...
        ```
    """
    cx, cy, cz = center
    start_angle_rad = np.deg2rad(start_angle_deg)
    end_angle_rad = np.deg2rad(end_angle_deg)

    # Calculate angles for each segment
    angles = np.linspace(start_angle_rad, end_angle_rad, num_segments + 1)

    wall_ids = []

    for i in range(num_segments):
        # Calculate midpoint angle for this segment
        mid_angle = (angles[i] + angles[i + 1]) / 2

        # Calculate segment length (arc length)
        arc_length = radius * (angles[i + 1] - angles[i])

        # Position at the midpoint of the arc
        x = cx + radius * np.cos(mid_angle)
        y = cy + radius * np.sin(mid_angle)
        z = cz

        # Create collision shape
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arc_length / 2, thickness / 2, height / 2]
        )

        # Create visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[arc_length / 2, thickness / 2, height / 2],
            rgbaColor=color
        )

        # Calculate orientation (perpendicular to radius)
        orientation_angle = mid_angle
        orientation_quat = p.getQuaternionFromEuler([0, 0, orientation_angle])

        # Create the box segment
        box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z],
            baseOrientation=orientation_quat
        )

        wall_ids.append(box_id)

    return wall_ids


def track_trajectory(position, prev_position=None, line_width=0.5):
    """
    Tracks and visualizes the drone's trajectory by drawing a line from the previous
    position to the current position.

    Parameters:
        position (Sequence[float]): Current 3D position [x, y, z] of the drone.
        prev_position (Sequence[float], optional): Previous 3D position [x, y, z].
            If None, no line is drawn (use for initialization). Defaults to None.
        color (Sequence[float], optional): RGB color of the trajectory line.
            Defaults to (0, 1, 0) (green).
        line_width (float, optional): Width of the trajectory line. Defaults to 2.

    Returns:
        int or None: The ID of the created debug line, or None if prev_position is None.

    How to:
        Before the simulation loop, initialize the trajectory tracker:
        ```
        ...
        prev_pos = None

        for i in range(simulation_seconds):
            ...
        ```
        Inside the simulation loop, after getting the current position:
        ```
        ...
        for i in range(simulation_seconds):
            current_pos = test_env.pos[0]  # Get current position of first drone

            # Draw trajectory line
            track_trajectory(current_pos, prev_pos, color=(0, 1, 0), line_width=2)

            # Update previous position
            prev_pos = current_pos.copy()
            ...
        ```
    """
    if prev_position is not None:
        line_id = p.addUserDebugLine(
            lineFromXYZ=prev_position,
            lineToXYZ=position,
            lineColorRGB=[0.25, 0.25, 0.25],
            lineWidth=line_width,
            lifeTime=0  # 0 means the line persists indefinitely
        )
        return line_id
    return None


def compute_delay_steps(delay_seconds: float, ctrl_freq_hz: float) -> int:
    return max(0, int(round(delay_seconds * ctrl_freq_hz)))


def make_obs_delay_buffer(initial_obs: np.ndarray, delay_steps: int) -> Deque[np.ndarray]:
    return deque([initial_obs.copy() for _ in range(delay_steps + 1)], maxlen=delay_steps + 1)


def feed_and_get_delayed(buffer: Deque[np.ndarray], latest_obs: np.ndarray) -> np.ndarray:
    buffer.append(latest_obs.copy())
    return buffer[0]


def unit_from_bearing(direction_deg: float) -> np.ndarray:
    rad = np.deg2rad(direction_deg)
    return np.array([np.cos(rad), np.sin(rad), 0.0], dtype=float)


def wind_force_from_speed(
        speed_mps: float,
        air_density: float,
        area_m2: float,
        drag_cd: float,
        direction_deg: float
) -> np.ndarray:
    magnitude = 0.5 * air_density * area_m2 * (speed_mps ** 2) * drag_cd
    return magnitude * unit_from_bearing(direction_deg)


def ou_next(
        value: float,
        dt: float,
        mean: float,
        theta: float,
        sigma: float,
        rng: np.random.Generator
) -> float:
    return value + theta * (mean - value) * dt + sigma * np.sqrt(dt) * rng.normal()


def make_wind_updater(
        mean_speed_mps: float,
        mean_dir_deg: float,
        theta: float,
        sigma_speed: float,
        sigma_dir: float,
        air_density: float,
        area_m2: float,
        drag_cd: float,
        dt: float,
        seed: int | None = None
) -> Callable[[], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    speed = max(0.0, mean_speed_mps)
    direction = mean_dir_deg % 360.0

    def update() -> np.ndarray:
        nonlocal speed, direction
        speed = max(0.0, ou_next(speed, dt, mean_speed_mps, theta, sigma_speed, rng))
        direction = (ou_next(direction, dt, mean_dir_deg, theta, sigma_dir, rng)) % 360.0
        return wind_force_from_speed(speed, air_density, area_m2, drag_cd, direction)

    return update