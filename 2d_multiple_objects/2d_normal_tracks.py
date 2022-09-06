from copy import deepcopy
from math import dist
import numpy as np
import plotly.graph_objects as go

np.random.seed(4)


def direction_match(step):
    value = 1
    if step == "LEFT":
        return [-value, 0]
    elif step == "RIGHT":
        return [value, 0]
    elif step == "UP":
        return [0, value]
    else:
        return [0, -value]


def spiral_match(step):
    value = 0.5
    norm = 1
    if step == "LEFT":
        return [-value, value/norm]
    elif step == "RIGHT":
        return [value, value/norm]
    elif step == "UP":
        return [0, value/1.1]
    else:
        return [0, -value]


def track_random_walk(time_steps, p, start=0):
    actual_track = []
    x_axis = np.array([start])
    y_axis = np.zeros(1)
    initial_data = np.concatenate((x_axis, y_axis))
    actual_track.append(initial_data)

    steps = np.random.choice(["LEFT", "RIGHT", "UP", "DOWN"], time_steps-1, p=p)
    for step in steps:
        temp_data = deepcopy(actual_track[-1])
        val = spiral_match(step)
        temp_data = temp_data + val
        actual_track.append(temp_data)

    actual_track = np.array(actual_track)
    actual_track = actual_track + (np.random.rand(actual_track.shape[0], actual_track.shape[1]) / 3)
    return actual_track


# Start of module
if __name__ == "__main__":
    time = 200
    reading = 2
    # sensor = random_walk_2d(num_sensors=1, time=time, reading=reading)
    fig = go.Figure()

    sen1 = track_random_walk(time, [0.05, 0.25, 0.69, 0.01]).reshape(time, 2, 1)
    # sen2 = track_random_walk(time, [0.15, 0.30, 0.54, 0.01], start=1).reshape(time, 2, 1)
    sen3 = track_random_walk(time, [0.30, 0.30, 0.39, 0.01], start=5).reshape(time, 2, 1)
    # sensor = np.concatenate((sen1, sen2, sen3), axis=2)

    sensor = np.concatenate((sen1, sen3, ), axis=2)
    time_ = np.arange(0, time)

    color = ['rgba(0, 0, 255, 1)', 'rgba(255, 0, 0, 1)', 'rgba(0, 2555, 0, 1)', 'rgba(0, 0, 0, 1)',
             'rgba(255, 255, 255, 1)', 'rgba(0, 0, 100, 1)', 'rgba(0, 100, 100, 1)', 'rgba(100, 0, 0, 1)',
             'rgba(100, 100, 0, 1)', 'rgba(255, 0, 100, 1)']

    # sensors = sensor.reshape(sensor.shape[0], 2)
    for i in range(reading):
        fig.add_trace(go.Scatter(x=[i[0] for i in sensor[:, :, i]], y=[i[1] for i in sensor[:, :, i]],
                                 mode='markers'
                                 # mode="lines + text", text=[str(j) for j in range(time)],
                                 # marker={'size': 9, 'color': color}
                                 ))

    # total_euclidean_dist = np.zeros(reading)
    # dist_hist = {}
    # track_hist = {}
    #
    # for i in range(time):
    #
    #     track_hist[i] = dict()
    #     dist_hist[i] = dict()
    #     current_euclidean_dist = []
    #     if i > 10:
    #
    #         for j in range(sensor.shape[2]):
    #             # z = 0 if i == 0 else i-1          # Absolute total distance
    #             z = 0 if i == 0 else 0              # Total distance from origin
    #             distance = dist((sensor[z, 0, j], sensor[z, 1, j]), (sensor[i, 0, j], sensor[i, 1, j]))
    #             current_euclidean_dist.append(distance)
    #
    #         current_euclidean_dist = np.array(current_euclidean_dist)
    #         total_euclidean_dist += current_euclidean_dist
    #
    #         track = np.argmax(total_euclidean_dist)
    #         track_hist.append(track)
    #
    #     else:
    #
    #         # Absolute total distance
    #         z = 0 if i == 0 else i - 1
    #         i = 1
    #
    #         # Total distance from origin
    #         # z = 0 if i == 0 else 0
    #
    #         for j in range(reading):
    #             current_euclidean_dist = []
    #             for k in range(reading):
    #                 # track_hist[i].append(sensor[i, :, j])
    #                 distance = dist((sensor[z, 0, j], sensor[z, 1, j]), (sensor[i, 0, k], sensor[i, 1, k]))
    #                 current_euclidean_dist.append(distance)
    #
    #             dist_hist[z][j] = current_euclidean_dist
    #         dist_hist[i] = None
    #
    #     # if dist_hist:
    #     #     print(f'Previous Track: {track_hist[-1]}, Current Track: {track}, with accumulative max distance: '
    #     #           f'{max(total_euclidean_dist)}')
    #     #
    #     # else:
    #     #     print(f'Previous_track: {0}, Detected Track: {track}, with accumulative max distance: '
    #     #           f'{max(total_euclidean_dist)}')
    #
    #     dist_hist.append([sensor[i, 0, track], sensor[i, 1, track]])
    #
    # track_x = [i[0] for i in dist_hist]
    # track_y = [i[1] for i in dist_hist]
    #
    # fig.add_trace(go.Scatter(x=track_x, y=track_y, mode="lines", marker={'size': 9, 'color': 'rgba(0, 0, 0, 1)'}))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(np.floor(np.min(sensor[:, 0, :])) - 1, np.ceil(np.max(sensor[:, 0, :]) + 1), dtype=int)

        ),
        # yaxis=dict(
        #     tickmode='array',
        #     tickvals=np.arange(np.floor(np.min(sensor[:, 1, :])) - 1, np.ceil(np.max(sensor[:, 1, :]) + 1), dtype=int)
        #
        # )
    )
    fig.show()
