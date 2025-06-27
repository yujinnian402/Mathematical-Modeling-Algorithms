import networkx as nx
import matplotlib.pyplot as plt

# 最短路可视化
def draw_shortest_path():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ('A', 'B', 2), ('A', 'C', 5), ('B', 'C', 1), ('B', 'D', 3), ('C', 'D', 2)
    ])
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

# 网络流可视化
def draw_network_flow():
    G = nx.DiGraph()
    G.add_weighted_edges_from([
        ('S', 'A', 10), ('S', 'B', 5), ('A', 'B', 15), ('A', 'T', 10), ('B', 'T', 10)
    ])
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightgreen')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()

if __name__ == '__main__':
    draw_shortest_path()
    draw_network_flow()
