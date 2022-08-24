from copy import deepcopy
import numpy as np
import plotly.graph_objects as go


# np.random.seed(50)


# def GroundTruths(num_sensors=1, time=10, reading=10):
#     num_sensors = num_sensors
#     # initial = np.ones(time, reading)
#     sensor_data = np.empty((1, reading))
#     for i in range(time):
#         prob = [0.60, 0.40]
#         start = 2
#         positions = [start]
#         random_points = np.random.random(reading)
#         down_points = random_points < prob[0]
#         up_points = random_points > prob[1]
#
#         for idown, iupp in zip(down_points, up_points):
#             down = idown and positions[-1] > 1
#             up = iupp and positions[-1] < 4
#             positions.append(positions[-1] - down + up)
#
#         sensor_data = np.concatenate((sensor_data, np.array(positions[:-1]).reshape(1, -1)), axis=0)
#     sensor_data = sensor_data[1:, :].T
#     sensor_data = sensor_data + np.random.rand(sensor_data.shape[0], sensor_data.shape[1])
#     return sensor_data


def random_walk(num_sensors=1, time=10, reading=10):
    num_sensors = num_sensors
    # initial = np.ones(time, reading)
    sensor_data = []
    initial_data = np.zeros((1, reading))
    sensor_data.append(deepcopy(np.ravel(initial_data).tolist()))
    for _ in range(time):
        prob = [0.50, ]
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
    time = 10
    sensor = random_walk(num_sensors=1, time=time, reading=10)
    # data = pd.DataFrame(sensor)
    fig = go.Figure()

    # sensor[:, 5] = np.linspace(-1, 1.8, sensor.shape[0])
    sensor[:, 5] = np.arange(0, sensor.shape[0])
    time = np.arange(0, time)

    for i in range(10):
        fig.add_trace(go.Scatter(x=time, y=sensor[i, :], mode="markers + text", text=[str(i) for _ in range(10)],
                                 marker={'size': 9}))

    diff = {}
    for i in range(sensor.shape[0] - 1):
        abs_diff = np.abs(sensor[i + 1, :] - sensor[i, :])
        # print(f'Minimum difference: {np.argmin(abs_diff)}')
        print(f'Minimum difference: {np.argmin(abs_diff)}, with distance: {min(abs_diff)}')
        # print(f"sorted array is : {np.sort(abs_diff)}")
        diff[i] = abs_diff

    fig.show()
