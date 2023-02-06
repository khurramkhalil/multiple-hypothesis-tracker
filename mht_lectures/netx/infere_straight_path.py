# Import libraies
import math

import networkx as nx
import numpy as np


# Generate crossing path
def x_align(length, offset):
    x_vals = np.linspace(0, 10, length)
    y_vals = np.repeat(offset, length)

    reading = np.vstack((x_vals, y_vals))
    reading = reading.T

    return reading


if __name__ == "__main__":
    duration = 5
    dims = 2
    num_targets = 2

    # Initialize graph object
    G = nx.Graph()

    # As targets are known, we initialize two local hypothesis
    target_0 = x_align(duration, 2)
    target_1 = x_align(duration, 5)

    # Now, iterate through the detections and update local hypothesis based on shortest distance
    for time_, j in enumerate(zip(target_0, target_1)):
        if time_ == 0:
            G.add_node(0, weight=1, loc=j[0].tolist())
            G.add_node(1, weight=1, loc=j[1].tolist())
            # G.add_nodes_from(range(num_targets))

        else:
            # First detected target
            G.add_node(len(G.nodes), weight=1, loc=j[0])
            diff_0 = math.dist(j[0], G.nodes[len(G.nodes) - 3]['loc'])
            diff_1 = math.dist(j[0], G.nodes[len(G.nodes) - 2]['loc'])

            if diff_0 < diff_1:
                G.add_edge(len(G.nodes) - 1, len(G.nodes) - 3)
            else:
                G.add_edge(len(G.nodes) - 1, len(G.nodes) - 2)

            # Second detected target
            G.add_node(len(G.nodes), weight=1, loc=j[1])
            diff_0 = math.dist(j[1], G.nodes[len(G.nodes) - 3]['loc'])
            diff_1 = math.dist(j[1], G.nodes[len(G.nodes) - 2]['loc'])

            if diff_0 < diff_1:
                G.add_edge(len(G.nodes) - 1, len(G.nodes) - 3)
            else:
                G.add_edge(len(G.nodes) - 1, len(G.nodes) - 2)

    nx.draw(G)
    print(time_, j)
