from copy import deepcopy
import numpy as np
import plotly.graph_objects as go


# np.random.seed(50)


def random_walk(num_sensors=1, time=10, reading=10):
    # Data accumulation list
    sensor_data = []
    # Initial values
    initial_data = np.zeros((1, reading))
    sensor_data.append(deepcopy(np.ravel(initial_data).tolist()))

    for _ in range(time):
        prob = [0.60, ]
        prob_array = np.random.random(reading)

        step_up = prob_array < prob[0]
        step_down = prob_array >= prob[0]

        initial_data[-1, step_up] += 1
        initial_data[-1, step_down] -= 1
        sensor_data.append(deepcopy(np.ravel(initial_data).tolist()))

    sensor_data = np.array(sensor_data)
    sensor_data = sensor_data + (np.random.rand(sensor_data.shape[0], sensor_data.shape[1]) / 5)
    return sensor_data


# Start of module
if __name__ == "__main__":
    time = 20
    reading = 30
    sensor = random_walk(num_sensors=1, time=time, reading=reading)
    # data = pd.DataFrame(sensor)
    fig = go.Figure()

    # sensor[:, 5] = np.linspace(-1, 1.8, sensor.shape[0])
    sensor[:, 5] = np.arange(0, sensor.shape[0]) + (np.random.rand(sensor.shape[0]) / 5)
    time_ = np.arange(0, reading)

    for i in range(time):
        fig.add_trace(go.Scatter(x=time_, y=sensor[i, :], mode="markers + text", text=[str(i) for _ in range(reading)],
                                 marker={'size': 9}))

    total_dist = np.zeros(sensor.shape[1])
    dist_hist = []
    for i in range(time - 1):
        abs_diff = sensor[i+1, :] - 0
        total_dist += abs_diff

        track = np.argmax(total_dist)
        print(f'Detected Track: {track}, with accumulative max distance: {max(total_dist)}')
        # print(f'Minimum difference: {np.argmax(abs_diff)}, with distance: {max(abs_diff)}')

        dist_hist.append([track, abs_diff[track]])

    track_y = [i[1] for i in dist_hist]
    track_y.insert(0, 0)

    track_x = [i[0] for i in dist_hist]
    track_x.insert(0, track_x[0])

    fig.add_trace(go.Scatter(x=track_x, y=track_y, mode="lines"))

    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, sensor.shape[1], dtype=int)

        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(np.floor(sensor.min()), sensor.max(), dtype=int)

        )
    )
    fig.show()
