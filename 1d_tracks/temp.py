from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
np.random.seed(50)

def GroundTruth(num_sensors=1, time=100, reading=10):
    # # Log curve
    # sensor_data = np.log(np.linspace(0, 10, length))
    # # Exponential curve
    # sensor_data = np.exp(np.linspace(0, 5, length))
    # sensor_data = 4 * (sensor_data**3) + 2 * (sensor_data**2) + 5 * sensor_data
    # Straight line
    # sensor_data = np.linspace(0, 10, length).reshape(-1, 1)

    sensor_data = np.random.rand(time, reading) * 2

    if num_sensors > 1:
        sensor_data = np.random.randn(time, reading, num_sensors)
        # sensor_data = np.repeat(sensor_data, num_sensors, axis=1)
    #
    # start_time = timedelta(seconds=0)
    # end_time = timedelta(seconds=length)
    # total_time = np.arange(start_time.seconds, end_time.seconds)
    #
    # for step in np.nditer(total_time):
    #     yield step, sensor_data[step, :]

    return sensor_data


if __name__ == "__main__":
    time = 10
    sensor = GroundTruth(num_sensors=1, time=time, reading=10)
    # data = pd.DataFrame(sensor)
    fig = go.Figure()
    # for i in range(time):
    #     # sensor[i, :] = i
    #     # fig.add_trace(go.Scatter(x=sensor[i, :], y=[i], mode="markers", ))
    #     fig.add_trace(go.Scatter(x=[1,2,3,4,5,6,7,8,9], y=[1,2,3,4,5,6,7,8,9], mode="markers", ))
    #     print(i)

    sensor[:, 5] = np.linspace(-1, 1, 10)
    time = np.arange(0, time)
    # fig.add_trace(go.Scatter(x=time, y=[np.random.rand() for i in time], mode="markers + text",))
    # fig.add_trace(go.Scatter(x=time, y=[np.random.randn() for i in time], mode="markers",))
    # fig.add_trace(go.Scatter(x=time, y=[np.random.rand() for i in time], mode="markers",))
    # fig.add_trace(go.Scatter(x=time, y=[np.random.rand() for i in time], mode="markers",))
    # fig.add_trace(go.Scatter(x=time, y=[np.random.randn() for i in time], mode="markers",))

    fig.add_trace(go.Scatter(x=time, y=sensor[0, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[1, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[2, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[3, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[4, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[5, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[6, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[7, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[8, :], mode="markers",))
    fig.add_trace(go.Scatter(x=time, y=sensor[9, :], mode="markers",))
    # fig.add_trace(go.Scatter(x=sensor, y=[i], mode="markers", ))
    # diff = {}
    # for i in range(sensor.shape[0] - 1):
    #     abs_diff = np.abs(sensor[i+1, :] - sensor[i, :])
    #     diff[i] =

    fig.show()
