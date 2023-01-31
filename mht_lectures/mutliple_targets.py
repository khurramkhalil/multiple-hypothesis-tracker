import numpy as np
import plotly.graph_objects as go


def x_align(length):
    x_vals = np.linspace(0, 10, length)
    y_vals = np.repeat(2, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


def y_align(length):
    y_vals = np.linspace(0, 10, length)
    x_vals = np.repeat(2, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


def straight_path(length):
    x_vals = np.linspace(0, 10, length)
    y_vals = np.linspace(0, 10, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


def exponential(length):
    x_vals = np.linspace(0, 10, length)
    y_vals = np.exp(np.linspace(0, 5, length))

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


def log_path(length):
    x_vals = np.linspace(0, 10, length)
    y_vals = np.log(np.linspace(0, 5, length))

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


if __name__ == "__main__":
    duration = 100
    dims = 2
    num_targets = 3
    possible_track_trajectories = [x_align, y_align, straight_path, exponential, log_path]
    paths = np.empty((0, dims))
    for target in range(num_targets):

        target_path = np.random.choice(possible_track_trajectories)
        path = target_path(duration)
        paths = np.append(paths, path, axis=0)

    # Separate each axis data
    x_coord = [x[0] for x in paths]
    y_coord = [x[1] for x in paths]

    # Initialize Plotly Fig object
    fig = go.Figure()
    # Add scatter plot for the sensor data
    fig.add_trace(go.Scatter(x=x_coord, y=y_coord, mode="markers", ))
    fig.show()
