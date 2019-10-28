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

def load_data(path):
    with open(path) as f:
        data = [[int(x) for x in line.split()] for line in f]
    data = np.array(data)
    return(data)

def individuals(data):
    return(np.unique(data[:,1:]))



def plot_normalized_weights_degree(GG):
    nodes_weights = []
    degrees = []
    for node in GG.nodes():
        degrees.append(GG.degree(node))
        links = GG.edges(node)
        node_weights = 0
        for i in links:
            node_weights = node_weights + GG[i[0]][i[1]]["weight"]

        nodes_weights.append(node_weights)
        
    nodes_weights = np.array(nodes_weights)/np.max(nodes_weights)
    degrees = np.array(degrees)/np.max(degrees)
        
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.title("Weights")
    plt.hist(nodes_weights,bins=50)
    plt.subplot(122)
    plt.title("Degree")
    plt.hist(degrees,bins=50)
    plt.show()
    
