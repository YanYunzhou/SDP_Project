# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import ex2
import ex2 as ex
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def import_dataset():
    filename = 'data/intel.g2o'
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
        node_translation=node_pose[0:2]
        node_angle=node_pose[2]
    for edge in edges:
        edge_type=edge[0]
        fromNode=edge[1]
        toNode=edge[2]
        measurement=edge[3]
        information=edge[4]
        translation=measurement[0:2]
        angle=measurement[2]
        rotation_matrix=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
def stack_p_r(nodes,x,lut):
    p_stack=np.array([])
    r_stack=np.array([])
    for node in nodes:
        pose=x[lut[node]:lut[node]+3]
        translation=pose[0:2]
        angle=pose[2]
        angle_v=angle_to_vec(angle)
        p_stack=np.append(p_stack,translation)
        r_stack=np.append(r_stack,angle_v)
    return p_stack,r_stack
def compute_loss():
    filename = 'data/intel.g2o'
    graph = ex.read_graph_g2o(filename)
    nodes = graph.nodes
    edges = graph.edges
    x = graph.x
    lut = graph.lut
    A=compute_incidence_matrix(edges,nodes)
    A_kron=compute_incidence_kron(A)
    p_stack, r_stack = stack_p_r(nodes, x,lut)
    D_block =compute_block_D(edges,nodes)
    loss=loss_function_3(A_kron,D_block,p_stack,r_stack)
    print(loss)
    loss=loss_function_1(x,edges,lut)
    print(loss)
def compute_incidence_matrix(edges,nodes):
    edge_num=len(edges)
    node_num=len(nodes)
    A=np.zeros((edge_num,node_num))
    for i in range(edge_num):
        fromNode = edges[i][1]
        toNode = edges[i][2]
        A[i,fromNode]=-1
        A[i,toNode]=1
    return A
def compute_incidence_kron(A):
    I=np.identity(2)
    A_kron=np.kron(A,I)
    return A_kron
def angle_to_mat(angle):
    rotation_mat=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    return rotation_mat
def angle_to_vec(angle):
    rotation_vec=np.array([math.cos(angle),math.sin(angle)])
    return rotation_vec
def t2D(delta):
    D=np.array([[delta[0],-delta[1]],[delta[1],delta[0]]])
    return D
def compute_block_D(edges,nodes):
    edge_num = len(edges)
    node_num=len(nodes)
    D_block=np.zeros((2*edge_num,2*node_num))
    edge_index=0
    for edge in edges:
        edge_type = edge[0]
        fromNode = edge[1]
        toNode = edge[2]
        measurement = edge[3]
        translation = measurement[0:2]
        D = t2D(translation)
        D_block[2*edge_index:2*edge_index+2,2*fromNode:2*fromNode+2]=-D
        edge_index=edge_index+1
    return D_block
def loss_function_1(x,edges,lut):
    loss=0
    for edge in edges:
        edge_type=edge[0]
        fromNode=edge[1]
        toNode=edge[2]
        measurement=edge[3]
        information=edge[4]
        information=np.diag(information)
        information=information/np.linalg.norm(information)
        infomration_v=(information[0]+information[1])/2
        infomration_w=information[2]
        translation=measurement[0:2]
        angle=measurement[2]
        rotation_matrix=angle_to_mat(angle)
        from_pose=x[lut[fromNode]:lut[fromNode]+3]
        from_translation=from_pose[0:2]
        from_rotation=angle_to_mat(from_pose[2])
        to_pose = x[lut[toNode]:lut[toNode]+3]
        to_translation = to_pose[0:2]
        to_rotation = angle_to_mat(to_pose[2])
        loss=loss+np.linalg.norm(to_translation-from_translation-np.matmul(from_rotation,translation))*np.linalg.norm(to_translation-from_translation-np.matmul(from_rotation,translation))
        loss=loss+0.5*np.linalg.norm(to_rotation-np.matmul(from_rotation,rotation_matrix),ord='fro')*np.linalg.norm(to_rotation-np.matmul(from_rotation,rotation_matrix),ord='fro')

    return loss
def loss_function_2(x,edges,lut):
    loss = 0
    for edge in edges:
        edge_type = edge[0]
        fromNode = edge[1]
        toNode = edge[2]
        measurement = edge[3]
        information = edge[4]
        information = np.diag(information)
        information = information / np.linalg.norm(information)
        translation = measurement[0:2]
        D=t2D(translation)
        angle = measurement[2]
        rotation_matrix=angle_to_mat(angle)
        from_pose = x[lut[fromNode]:lut[fromNode] + 3]
        from_translation = from_pose[0:2]
        from_rotation = angle_to_vec(from_pose[2])
        to_pose = x[lut[toNode]:lut[toNode] + 3]
        to_translation = to_pose[0:2]
        to_rotation = angle_to_vec(to_pose[2])
        loss = loss + np.linalg.norm(
            to_translation - from_translation - np.matmul(D,from_rotation)) * np.linalg.norm(
            to_translation - from_translation - np.matmul(D,from_rotation))
        loss = loss +  np.linalg.norm(to_rotation - np.matmul(rotation_matrix, from_rotation)) * np.linalg.norm(to_rotation - np.matmul(rotation_matrix, from_rotation))

    return
def loss_function_3(A_kron,D_block,p_stack,r_stack):
    loss=np.linalg.norm(np.matmul(A_kron,p_stack)+np.matmul(D_block,r_stack))*np.linalg.norm(np.matmul(A_kron,p_stack)+np.matmul(D_block,r_stack))
    return loss
def plot_graph():
    filename = 'data/intel.g2o'
    graph = ex.read_graph_g2o(filename)
    ex2.plot_graph(graph)

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compute_loss()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
