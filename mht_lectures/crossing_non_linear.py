import numpy as np
import plotly.graph_objects as go


def curve(length):
    poly_coefs = [4, 3, 2, 1]

    x_vals = np.linspace(-40, 40, length)
    y_vals = np.polyval(poly_coefs, x_vals) / 3000

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


def flip_curve(length):
    poly_coefs = [4, 3, 2, 1]

    x_vals = np.linspace(-40, 40, length)
    y_vals = np.polyval(poly_coefs, x_vals) / 3000  # - 10
    x_vals = np.linspace(40, -40, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


if __name__ == "__main__":
    duration = 100
    dims = 2
    possible_track_trajectories = [curve, flip_curve]
    paths = np.empty((0, dims))
    for target_path in possible_track_trajectories:
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
