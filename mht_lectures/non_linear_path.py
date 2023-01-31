from datetime import timedelta

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
    y_vals = np.polyval(poly_coefs, x_vals) / 3000
    x_vals = np.linspace(40, -40, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


if __name__ == "__main__":
    duration = 100

    path = flip_curve(duration)
    for i in path:
        print(f'x coordinate, {i[0]}, y coordinate: {i[1]}')

    # Separate each axis data
    x_coord = [x[0] for x in path]
    y_coord = [x[1] for x in path]

    # Initialize Plotly Fig object
    fig = go.Figure()
    # Add scatter plot for the sensor data
    fig.add_trace(go.Scatter(x=x_coord, y=y_coord, mode="markers", ))
    fig.show()
