import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            tmp = line.split()[0:3]
            arr_tmp = [int(tmp[0]),int(tmp[1]),int(tmp[2])]
        
            data.append(arr_tmp)
    data = np.array(data)
    return(data)


def individuals(data):
    return(np.unique(data[:,1:]))



def build_weighted_graph(data,gap=19):
    nodes = individuals(data)
    graphs,pos = build_graphs(data,gap)

    G = nx.complete_graph(nodes)
    for e in G.edges():
        G[e[0]][e[1]]["weight"]=0

    for g in graphs:
        for e in g.edges():
            G[e[0]][e[1]]["weight"] = G[e[0]][e[1]]["weight"] + 1
    GG = G.copy()
    for e in G.edges():
        if (G[e[0]][e[1]]["weight"] == 0):
            GG.remove_edge(e[0],e[1])
            
    return(GG)


def not_consecutive_duplicates_data(data):
    i_old = 0
    j_old = 0
    new_data = []
    for t,i,j in data:
        if not(i == i_old and j==j_old):
            i_old = i
            j_old = j
            new_data.append([t,i,j])
    new_data = np.array(new_data)
    return(new_data)
    
def build_weighted_graph_2(data,gap=19):
    nodes = individuals(data)
    data = not_consecutive_duplicates_data(data)
    graphs,pos = build_graphs(data,gap)

    G = nx.complete_graph(nodes)
    for e in G.edges():
        G[e[0]][e[1]]["weight"]=0

    for g in graphs:
        for e in g.edges():
            G[e[0]][e[1]]["weight"] = G[e[0]][e[1]]["weight"] + 1
    GG = G.copy()
    for e in G.edges():
        if (G[e[0]][e[1]]["weight"] == 0):
            GG.remove_edge(e[0],e[1])
            
    return(GG)



def build_graphs(data,gap=19):
    graphs = []
    G=nx.Graph()
    nodes = individuals(data)
    G.add_nodes_from(nodes)
    pos = nx.spring_layout(G)
    splitted_data = split_input_data(data,gap)
    
    for t in splitted_data:
        g = G.copy()
        for _,i,j in t:
            g.add_edge(i,j)
        graphs.append(g)  

    return(graphs,pos)




def split_input_data(data, gap=19):
    times = data[:,0]
    pos = times[0]
    chunks = []
    for i in range(len(times)):
        if not(times[i]<=(pos + gap)):
            chunks.append(i)
            pos = times[i]

    return(np.split(data,chunks))

