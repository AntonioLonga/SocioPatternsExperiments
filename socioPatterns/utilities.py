import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_weighted_graph(GG,pos, node_size=70):
    nx.draw_networkx_nodes(GG, pos, node_size=node_size)

    # edges
    for u,v in GG.edges():
        nx.draw_networkx_edges(GG,pos, edgelist=[(u,v)],width=int(GG[u][v]["weight"]))

    plt.axis('off')
    plt.show()

def build_weighted_graph(data):
    nodes = individuals(data)
    G = nx.complete_graph(nodes)
    for e in G.edges():
        G[e[0]][e[1]]["weight"]=0
    for t,i,j in data:
        G[i][j]["weight"] = G[i][j]["weight"] + 1

    GG = G.copy()
    for e in G.edges():
        if (G[e[0]][e[1]]["weight"] == 0):
            GG.remove_edge(e[0],e[1])
    return(GG)

def plot_graphs(graphs,splitted_data,pos):
    
    c=0
    interaction = [len(i) for i in splitted_data]
    for i in graphs:
        plt.figure(figsize=(10,10))
        plt.subplot(221)
        nx.draw(i,pos=pos, node_size=1,with_labels=False)
        plt.subplot(222)
        barlist = plt.bar(np.arange(len(interaction)),interaction)
        barlist[c].set_color('r')
        plt.show()
        c = c+1
        
def build_graphs(data,gap):
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

def split_input_data(data, gap):
    times = data[:,0]
    pos = times[0]
    chunks = []
    for i in range(len(times)):
        if not(times[i]<=(pos + gap)):
            chunks.append(i)
            pos = times[i]

    return(np.split(data,chunks))

def load_data(path):
    with open(path) as f:
        data = [[int(x) for x in line.split()] for line in f]
    data = np.array(data)
    return(data)
def individuals(data):
    return(np.unique(data[:,1:]))