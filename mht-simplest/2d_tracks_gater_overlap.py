from math import dist

import numpy as np
import networkx as nx
import plotly.graph_objects as go
from shapely.geometry import Point

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
    ground_truth_2 = curve(time_steps, upper_limits, lower_limits, 3000 - 10)[::-1]
    sen1 = normal_noise(ground_truth_1).reshape(time_steps, 2, 1)
    sen2 = normal_noise(ground_truth_2).reshape(time_steps, 2, 1)

    sensor = np.concatenate((sen1, sen2,), axis=2)
    time_ = np.arange(0, time_steps)

    color = ['rgba(0, 0, 255, 0.6)', 'rgba(255, 0, 0, 0.6)', 'rgba(0, 2555, 0, 1)', 'rgba(0, 0, 0, 1)',
             'rgba(255, 255, 255, 1)', 'rgba(0, 0, 100, 1)', 'rgba(0, 100, 100, 1)', 'rgba(100, 0, 0, 1)',
             'rgba(100, 100, 0, 1)', 'rgba(255, 0, 100, 1)']

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

    G = nx.Graph(tracker='2 tracks')
    gater_record = {0: [], 1: []}
    circumference_record = {0: [], 1: []}
    sensor_points_record = {}
    sensor_readings_record = {}
    for i in range(time_steps):
        # G.add_node[
        sensor_points = []
        reading_points = []
        for j in range(sensor.shape[2]):

            reading = sensor[i, :, j]
            point = Point(reading)
            gater = point.buffer(5)

            circumference = list(gater.exterior.coords)
            circumference.append([None, None])
            circumference_record[j].append(circumference)

            gater_record[j].append(gater)
            sensor_points.append(point)
            reading_points.append(reading.tolist())

            if i > 1:
                contains = [gater.contains(b) for b in sensor_points_record[i-1]]
                if all(contains):
                    node_val = sensor_readings_record[i-1]
                    # distance = dist((sensor[z, 0, j], sensor[z, 1, j]), (sensor[i, 0, k], sensor[i, 1, k]))
                    if node_val[-1][0]:
                        node_val.append([None, None])
                    flat_list = [item for sublist in node_val for item in sublist]
                    G.add_nodes_from([(i, {j: node_val})])

                else:
                    G.add_nodes_from([(i, {j: reading.tolist()})])

            else:
                G.add_nodes_from([(i, {j: reading.tolist()})])

        sensor_points_record[i] = sensor_points
        sensor_readings_record[i] = reading_points

    for i in circumference_record.keys():
        flat_list = [item for sublist in circumference_record[i] for item in sublist]
        fig.add_trace(go.Scatter(x=[x[0] for x in flat_list], y=[x[1] for x in flat_list], mode='lines'))

    track_0 = []
    track_1 = []
    for node in G.nodes:
        val = G.nodes[node][0]
        if len(val) > 2:
            for a in val:
                track_0.append(a)
                # track_1.append(G.nodes[node][1])

        else:
            track_0.append(G.nodes[node][0])
            track_1.append(G.nodes[node][1])

    fig.add_trace(go.Scatter(x=[i[0] for i in track_0], y=[i[1] for i in track_0], mode='lines'))
    fig.add_trace(go.Scatter(x=[i[0] for i in track_1], y=[i[1] for i in track_1], mode='lines'))

    fig.update_layout(
        yaxis_range=[np.min(sensor) - 10, np.max(sensor) + 10],
        xaxis_range=[np.min(sensor) - 10, np.max(sensor) + 10],
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
