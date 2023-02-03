import matplotlib.pyplot as plt
import networkx as nx

# Create an empty graph
G = nx.Graph()

# Add nodes to the graph
G.add_node(1)
G.add_node(2)
G.add_node(3)

# Add edges to the graph
G.add_edge(1, 2)
G.add_edge(2, 3)

# Check number of nodes and edges in the graph
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Get a list of all nodes in the graph
print("Nodes:", G.nodes())

# Get a list of all edges in the graph
print("Edges:", G.edges())

# Check if node 4 is in the graph
print("Node 4 in graph:", 4 in G)

# Remove node 3 from the graph
G.remove_node(3)

# Check number of nodes and edges in the graph after removing node 3
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# Draw the graph
plt.figure()
nx.draw(G)
plt.show()
