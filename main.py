# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math

import ex2
import ex2 as ex
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.sparse
def import_dataset_intel():
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
def simulation_dict():
    filename = 'data/simulation-pose-pose.g2o'
    graph = ex.read_graph_g2o(filename)
    nodes = graph.nodes
    edges = graph.edges
    x = graph.x
    lut = graph.lut
    nodes_num = len(nodes)
    edges_num = len(edges)
    lut_num = len(lut)
    node_count=0
    node_dict={}
    for nodeId in nodes:
        node=nodes[nodeId]
        node_dict[nodeId]=node_count
        node_count=node_count+1
    return node_dict
def vector_to_complex(vec):
    x=vec[0]
    y=vec[1]
    vec_norm=np.linalg.norm(vec)
    complex=(x+1j*y)/vec_norm
    return complex
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
def concat_pr(p_stack,r_stack):
    pr=np.append(p_stack,r_stack)
    return pr
def compute_pr_complex(P_stack):
    num=np.shape(P_stack)[0]/2
    num=int(num)
    p_complex=np.array([])
    for i in range(num):
        p_complex=np.append(p_complex,vector_to_complex(P_stack[2*i:2*i+2]))
    return p_complex
def compute_loss():
    filename = 'data/intel.g2o'
    graph = ex.read_graph_g2o(filename)
    nodes = graph.nodes
    edges = graph.edges
    x = graph.x
    lut = graph.lut
    A=compute_incidence_matrix(edges,nodes)
    A2=A[:,1:np.shape(A)[1]]
    A_kron=compute_incidence_kron(A)
    A_kron_2=compute_incidence_kron(A2)
    p_stack, r_stack = stack_p_r(nodes, x,lut)
    D_block =compute_block_D(edges,nodes)
    U=compute_U(edges,nodes)
    Q=np.matmul(np.transpose(D_block),D_block)+np.matmul(np.transpose(U),U)
    L=np.matmul(np.transpose(A_kron),A_kron)
    L2=np.matmul(np.transpose(A_kron_2),A_kron_2)
    W1=np.block([[L,np.matmul(np.transpose(A_kron),D_block)],[np.matmul(np.transpose(D_block),A_kron),Q]])
    W2=np.block([[L2,np.matmul(np.transpose(A_kron_2),D_block)],[np.matmul(np.transpose(D_block),A_kron_2),Q]])
    pr = concat_pr(p_stack, r_stack)
    pr_2=pr[2:np.shape(pr)[0]]
    loss=loss_function_4(W2,pr_2)
    print(loss)
    loss=loss_function_1(x,edges,lut)
    print(loss)
    pr_convex=compute_pr_complex(pr_2)
    W_complex=compute_W_complex(W2)
    loss = loss_function_5(W_complex, pr_convex)
    print(loss)
    eigen,eigen_vec=np.linalg.eig(W_complex)
    print(eigen)
    print(np.shape(eigen))
    for i in range(np.shape(eigen)[0]):
        print(eigen[i])


def compute_incidence_matrix(edges,nodes):
    edge_num=len(edges)
    node_num=len(nodes)
    A=np.zeros((edge_num,node_num))
    for i in range(edge_num):
        fromNode = edges[i][1]
        toNode = edges[i][2]
        if toNode<node_num:
            A[i, fromNode] = -1
            A[i, toNode] = 1
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
def compute_U(edges,nodes):
    edge_num = len(edges)
    node_num = len(nodes)
    U= np.zeros((2 * edge_num, 2 * node_num))
    edge_index = 0
    for edge in edges:
        edge_type = edge[0]
        fromNode = edge[1]
        toNode = edge[2]
        measurement = edge[3]
        translation = measurement[0:2]
        angle=measurement[2]
        rotation=angle_to_mat(angle)
        U[2*edge_index:2*edge_index+2,2*fromNode:2*fromNode+2]=-rotation
        U[2 * edge_index:2 * edge_index + 2, 2 * toNode:2 * toNode + 2]=np.identity(2)
        edge_index = edge_index + 1
    return U
def matrix_to_complex(mat):
    a=np.sqrt(mat[0,0]*mat[0,0]+mat[0,1]*mat[0,1])
    mat=mat/a
    complex_num=a*(mat[0,0]+1j*mat[1,0])
    return complex_num
def compute_W_complex(W):
    W_height=int(np.shape(W)[0]/2)
    W_width=int(np.shape(W)[1]/2)
    W_complex=np.zeros((W_height,W_width),dtype=complex)
    count=0
    for i in range(W_height):
        for j in range(W_width):
            w_complex=W[2*i:2*i+2,2*j:2*j+2]
            if np.linalg.det(w_complex)!=0:
                W_complex[i,j]=matrix_to_complex(w_complex)
                count=count+1
    return W_complex
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
def loss_function_3(A_kron,D_block,p_stack,r_stack,U):
    loss=np.linalg.norm(np.matmul(A_kron,p_stack)+np.matmul(D_block,r_stack))*np.linalg.norm(np.matmul(A_kron,p_stack)+np.matmul(D_block,r_stack))
    loss=loss+np.linalg.norm(np.matmul(U,r_stack))*np.linalg.norm(np.matmul(U,r_stack))
    return loss
def loss_function_4(W1,pr):
    loss=np.dot(pr,np.matmul(W1,pr))
    return loss
def loss_function_5(W_complex,p_complex):
    loss = np.dot(p_complex.conjugate(), np.matmul(W_complex, p_complex))
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
