from datetime import timedelta

import numpy as np
import plotly.graph_objects as go


def GroundTruth(num_sensors=1, length=100):
    # # Log curve
    # sensor_data = np.log(np.linspace(0, 10, length))
    # # Exponential curve
    # sensor_data = np.exp(np.linspace(0, 5, length))
    # sensor_data = 4 * (sensor_data**3) + 2 * (sensor_data**2) + 5 * sensor_data
    # Straight line
    sensor_data = np.linspace(0, 10, length).reshape(-1, 1)

    if num_sensors > 1:
        sensor_data = np.repeat(sensor_data, num_sensors, axis=1)

    start_time = timedelta(seconds=0)
    end_time = timedelta(seconds=length)
    total_time = np.arange(start_time.seconds, end_time.seconds)

    for step in np.nditer(total_time):
        yield step, sensor_data[step, :]


if __name__ == "__main__":
    sensor = GroundTruth(num_sensors=1, length=100)
    lst = []
    for i in sensor:
        lst.append(i)
        # print(f'Time, {i[0]}, reading: {i[1]}')

    time_ = [i[0] for i in lst]
    readings = [i[1][0] for i in lst]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_, y=readings, mode="markers",))
    fig.add_trace(go.Scatter(x=time_, y=[0.5 for i in lst], mode="lines", line={'dash': 'dash', 'color': 'green'}))
    fig.show()
    # fig.hold(True)
