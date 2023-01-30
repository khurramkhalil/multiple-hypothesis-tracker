from datetime import timedelta

import numpy as np
import plotly.graph_objects as go


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
    start_time = timedelta(seconds=0)
    end_time = timedelta(seconds=duration)
    total_time = np.arange(start_time.seconds, end_time.seconds)

    # x-align path
    path = log_path(duration)
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
