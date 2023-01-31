import numpy as np
import plotly.graph_objects as go

val = 5000


def curve(length):
    poly_coefs = [4, 3, 2, 1]

    x_vals = np.linspace(-40, 40, length)
    y_vals = np.polyval(poly_coefs, x_vals) / val

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


def flip_curve(length):
    poly_coefs = [4, 3, 2, 1]

    x_vals = np.linspace(-40, 40, length)
    y_vals = np.polyval(poly_coefs, x_vals) / val - 10
    x_vals = np.linspace(40, -40, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


if __name__ == "__main__":
    duration = 100
    dims = 2
    num_targets = 3
    possible_track_trajectories = [curve, flip_curve]
    paths = []
    for target_path in possible_track_trajectories:
        path = target_path(duration)
        paths.append(path)

    # Separate each axis data
    x_coord_0 = [x[0] + 40 for x in paths[0]]
    y_coord_0 = [x[1] + 60 for x in paths[0]]

    x_coord_1 = [x[0] + 40 for x in paths[1]]
    y_coord_1 = [x[1] + 60 for x in paths[1]]

    # Initialize Plotly Fig object
    fig = go.Figure()
    # Add scatter plot for the sensor data
    fig.add_trace(go.Scatter(x=x_coord_0, y=y_coord_0, mode="markers", ))
    fig.add_trace(go.Scatter(x=x_coord_0, y=y_coord_0, mode="lines", ))
    fig.add_trace(go.Scatter(x=x_coord_1, y=y_coord_1, mode="markers", ))
    fig.add_trace(go.Scatter(x=x_coord_1, y=y_coord_1, mode="lines", ))
    fig.show()
