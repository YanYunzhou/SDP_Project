# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ex2
import ex2 as ex
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def import_dataset():
    filename = 'data/dlr.g2o'
    graph = ex.read_graph_g2o(filename)
    nodes=graph.nodes
    edges=graph.edges
    x=graph.x
    lut=graph.lut
    nodes_num=len(nodes)
    edges_num=len(edges)
    lut_num=len(lut)
    for nodeId in nodes:
        dimension = len(nodes[nodeId])
        offset = lut[nodeId]
        node_pose=nodes[nodeId]
        if dimension==2:
            print("It is landmark!")
            print(node_pose)
        else:
            print("It is robot pose!")
            print(node_pose)
    for edge in edges:
        edge_type=edge[0]
        fromNode=edge[1]
        toNode=edge[2]
        measurement=edge[3]
        information=edge[4]
def plot_graph():
    filename = 'data/dlr.g2o'
    graph = ex.read_graph_g2o(filename)
    ex2.plot_graph(graph)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import_dataset()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
