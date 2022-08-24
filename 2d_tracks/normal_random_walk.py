from copy import deepcopy
from math import dist
import numpy as np
import plotly.graph_objects as go

np.random.seed(50)


def direction_match(step):
    if step == "LEFT":
        return [-1, 0]
    elif step == "RIGHT":
        return [1, 0]
    elif step == "UP":
        return [0, 1]
    else:
        return [0, -1]


def random_walk_2d(num_sensors, time, reading):
    sensor_data = []
    x_axis = np.arange(0, reading)
    y_axis = np.zeros(reading)
    initial_data = np.concatenate((x_axis, y_axis)).reshape(-1, 10)
    sensor_data.append(initial_data)

    for _ in range(time - 1):
        steps = np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"], reading)

        temp_data = deepcopy(sensor_data[-1])
        for idx, step in enumerate(steps):
            val = direction_match(step)
            temp_data[:, idx] = temp_data[:, idx] + val

        sensor_data.append(temp_data)

    sensor_data = np.array(sensor_data)
    # sensor_data = sensor_data + (np.random.rand(sensor_data.shape[0], sensor_data.shape[1]) / 5)
    return sensor_data


# Start of module
if __name__ == "__main__":
    time = 2
    reading = 10
    sensor = random_walk_2d(num_sensors=1, time=time, reading=reading)
    fig = go.Figure()

    # sensor[:, 5] = np.linspace(-1, 1.8, sensor.shape[0])
    # sensor[:, 5] = np.arange(0, sensor.shape[0])
    time_ = np.arange(0, time)
    color = ['rgba(0, 0, 255, 0.3)', 'rgba(0, 255, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(0, 0, 0, 0.3)'
             , 'rgba(255, 255, 255, 0.3)', 'rgba(0, 0, 100, 0.3)', 'rgba(0, 100, 100, 0.3)', 'rgba(100, 0, 0, 0.3)'
             , 'rgba(100, 100, 0, 0.3)', 'rgba(255, 0, 100, 0.3)']
    # color = ['rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(0, 0, 0, 1)',
    #          'rgba(255, 255, 255, 1)', 'rgba(0, 0, 100, 1)', 'rgba(0, 100, 100, 1)', 'rgba(100, 0, 0, 1)',
    #          'rgba(100, 100, 0, 1)', 'rgba(255, 0, 100, 1)']

    for i in range(time):
        fig.add_trace(go.Scatter(x=sensor[i, 0, :], y=sensor[i, 1, :], mode="markers + text",
                                 text=[str(j) for j in range(reading)],
                                 marker={'size': 9, 'color': color}))

    total_euclidean_dist = np.zeros(reading)
    dist_hist = []
    for i in range(sensor.shape[0]):
        current_euclidean_dist = []
        for j in range(sensor.shape[2]):
            distance = dist((sensor[0, 0, j], sensor[0, 1, j]), (sensor[i, 0, j], sensor[i, 1, j]))
            current_euclidean_dist.append(distance)

        current_euclidean_dist = np.array(current_euclidean_dist)
        total_euclidean_dist += current_euclidean_dist
        track = np.argmax(total_euclidean_dist)

        if dist_hist:
            print(
                f'Previous Track: {dist_hist[-1][0]}, Current Track: {track}, with accumulative max distance: '
                f'{max(total_euclidean_dist)}')

        else:
            print(
                f'Previous_track: {0}, Detected Track: {track}, with accumulative max distance: '
                f'{max(total_euclidean_dist)}')

        dist_hist.append([sensor[i, 0, track], sensor[i, 1, track]])

    track_x = [i[0] for i in dist_hist]
    track_y = [i[1] for i in dist_hist]

    fig.add_trace(go.Scatter(x=track_x, y=track_y, mode="lines"))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(np.floor(np.min(sensor[:, 0, :])) - 1, np.ceil(np.max(sensor[:, 0, :]) + 1), dtype=int)

        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(np.floor(np.min(sensor[:, 1, :])) - 1, np.ceil(np.max(sensor[:, 1, :]) + 1), dtype=int)

        )
    )
    fig.show()
