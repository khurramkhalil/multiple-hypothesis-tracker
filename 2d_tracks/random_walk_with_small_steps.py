from copy import deepcopy
from math import dist
import numpy as np
import plotly.graph_objects as go

# np.random.seed(50)


def direction_match(step):
    value = 0.1
    if step == "LEFT":
        return [-value, 0]
    elif step == "RIGHT":
        return [value, 0]
    elif step == "UP":
        return [0, value]
    else:
        return [0, -value]


def random_walk_2d(num_sensors, time, reading):
    sensor_data = []
    x_axis = np.arange(0, reading)
    y_axis = np.zeros(reading)
    initial_data = np.concatenate((x_axis, y_axis)).reshape(-1, reading)
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


def track_random_walk(time_steps, readings):
    actual_track = []
    x_axis = np.arange(0, 1)
    y_axis = np.zeros(1)
    initial_data = np.concatenate((x_axis, y_axis))
    actual_track.append(initial_data)

    steps = np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"], time_steps-1, p=[0.25, 0.25, 0.30, 0.20])
    for step in steps:
        temp_data = deepcopy(actual_track[-1])
        val = direction_match(step)
        temp_data = temp_data + val
        actual_track.append(temp_data)

    actual_track = np.array(actual_track)

    return actual_track


# Start of module
if __name__ == "__main__":
    time = 1000
    reading = 10
    sensor = random_walk_2d(num_sensors=1, time=time, reading=reading)
    fig = go.Figure()

    sensor[:, :, 5] = track_random_walk(time, reading)
    time_ = np.arange(0, time)
    # color = ['rgba(0, 0, 255, 0.3)', 'rgba(0, 255, 0, 0.3)', 'rgba(255, 0, 0, 0.3)', 'rgba(0, 0, 0, 0.3)'
    #          , 'rgba(255, 255, 255, 0.3)', 'rgba(0, 0, 100, 0.3)', 'rgba(0, 100, 100, 0.3)', 'rgba(100, 0, 0, 0.3)'
    #          , 'rgba(100, 100, 0, 0.3)', 'rgba(255, 0, 100, 0.3)']
    color = ['rgba(0, 0, 255, 1)', 'rgba(0, 255, 0, 1)', 'rgba(255, 0, 0, 1)', 'rgba(0, 0, 0, 1)',
             'rgba(255, 255, 255, 1)', 'rgba(0, 0, 100, 1)', 'rgba(0, 100, 100, 1)', 'rgba(100, 0, 0, 1)',
             'rgba(100, 100, 0, 1)', 'rgba(255, 0, 100, 1)']

    # sensors = sensor.reshape(sensor.shape[0], 2)
    for i in range(reading):
        fig.add_trace(go.Scatter(x=[i[0] for i in sensor[:, :, i]], y=[i[1] for i in sensor[:, :, i]],
                                 # mode="lines + text", text=[str(j) for j in range(time)],
                                 marker={'size': 9, 'color': color}))

    total_euclidean_dist = np.zeros(reading)
    dist_hist = []
    track_hist = []
    for i in range(sensor.shape[0]):
        current_euclidean_dist = []
        for j in range(sensor.shape[2]):
            # z = 0 if i == 0 else i-1          # Absolute total distance
            z = 0 if i == 0 else 0              # Total distance from origin
            distance = dist((sensor[z, 0, j], sensor[z, 1, j]), (sensor[i, 0, j], sensor[i, 1, j]))
            current_euclidean_dist.append(distance)

        current_euclidean_dist = np.array(current_euclidean_dist)
        total_euclidean_dist += current_euclidean_dist

        track = np.argmax(total_euclidean_dist)
        track_hist.append(track)

        # if dist_hist:
        #     print(f'Previous Track: {track_hist[-1]}, Current Track: {track}, with accumulative max distance: '
        #           f'{max(total_euclidean_dist)}')
        #
        # else:
        #     print(f'Previous_track: {0}, Detected Track: {track}, with accumulative max distance: '
        #           f'{max(total_euclidean_dist)}')

        dist_hist.append([sensor[i, 0, track], sensor[i, 1, track]])

    track_x = [i[0] for i in dist_hist]
    track_y = [i[1] for i in dist_hist]

    fig.add_trace(go.Scatter(x=track_x, y=track_y, mode="lines", marker={'size': 9, 'color': 'rgba(0, 0, 0, 1)'}))

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
