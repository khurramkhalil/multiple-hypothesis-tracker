from math import dist
import numpy as np
import plotly.graph_objects as go
import networkx as nx

np.random.seed(42)


def curve(num_pts, lower_limit, upper_limit, val=3000):
    polynomial = [1, 2, 3, 4, ]
    poly_coefs = polynomial[::-1]

    if lower_limit > upper_limit:
        x1 = np.linspace(lower_limit, upper_limit, num_pts)
        x = np.linspace(upper_limit, lower_limit, num_pts)
        y = np.polyval(poly_coefs, x) / val - 10
        return np.vstack((x1, y)).T

    else:
        x = np.linspace(lower_limit, upper_limit, num_pts)
        y = np.polyval(poly_coefs, x) / val

    return np.vstack((x, y)).T


def normal_noise(actual_track):
    actual_track = actual_track + (np.random.rand(actual_track.shape[0], actual_track.shape[1]) / 2)
    return actual_track


# Start of module
if __name__ == "__main__":
    time_steps = 150
    lower_limits = -40
    upper_limits = 40
    reading = 2

    ground_truth_1 = curve(time_steps, lower_limits, upper_limits, 3000)
    ground_truth_2 = curve(time_steps, upper_limits, lower_limits, 3000 - 10)
    sen1 = normal_noise(ground_truth_1).reshape(time_steps, 2, 1)
    sen2 = normal_noise(ground_truth_2).reshape(time_steps, 2, 1)
    # sensor = np.concatenate((sen1, sen2, sen3), axis=2)

    sensor = np.concatenate((sen1, sen2,), axis=2)
    time_ = np.arange(0, time_steps)

    color = ['rgba(0, 0, 255, 0.6)', 'rgba(255, 0, 0, 0.6)', 'rgba(0, 2555, 0, 1)', 'rgba(0, 0, 0, 1)',
             'rgba(255, 255, 255, 1)', 'rgba(0, 0, 100, 1)', 'rgba(0, 100, 100, 1)', 'rgba(100, 0, 0, 1)',
             'rgba(100, 100, 0, 1)', 'rgba(255, 0, 100, 1)']

    # sensors = sensor.reshape(sensor.shape[0], 2)
    fig = go.Figure()
    for i in range(reading):
        fig.add_trace(go.Scatter(x=[i[0] for i in sensor[:, :, i]], y=[i[1] for i in sensor[:, :, i]],
                                 mode='markers',
                                 # mode="lines + text", text=[str(j) for j in range(time)],
                                 marker={'color': color[i]},
                                 name=f"Measurements of Track {str(i)}"
                                 ))

    fig.add_trace(go.Scatter(x=[i[0] for i in ground_truth_1], y=[i[1] for i in ground_truth_1],
                             marker={'color': color[0]},
                             name="Ground Truth Track 0", mode='lines'))
    fig.add_trace(go.Scatter(x=[i[0] for i in ground_truth_2], y=[i[1] for i in ground_truth_2],
                             marker={'color': color[1]},
                             name="Ground Truth Track 1", mode='lines'))


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
        # xaxis=dict(
        #     tickmode='array',
        #     tickvals=np.arange(np.floor(np.min(sensor[:, 0, :])) - 1, np.ceil(np.max(sensor[:, 0, :]) + 1), dtype=int)
        #
        # ),
        # yaxis=dict(
        #     tickmode='array',
        #     tickvals=np.arange(np.floor(np.min(sensor[:, 1, :])) - 1, np.ceil(np.max(sensor[:, 1, :]) + 1), dtype=int)
        #
        # )
    )
    fig.show()
