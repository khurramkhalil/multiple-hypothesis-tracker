import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat


def main():
    num_frames = 100
    num_points = 10
    color_data = np.random.random((num_frames, num_points))
    x, y, c = np.random.random((3, num_points))

    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=range(num_frames), fargs=(color_data, scat))

    plt.show()


main()
