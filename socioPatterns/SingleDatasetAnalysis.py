from tabulate import tabulate
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utilities as ut
import construction as cs



def plot_weighted_graph(GG,pos, node_size=70):
    '''
    plot the final graphs showing where the size of the edge represent the weight on the edge
    '''
    nx.draw_networkx_nodes(GG, pos, node_size=node_size)

    # edges
    for u,v in GG.edges():
        nx.draw_networkx_edges(GG,pos, edgelist=[(u,v)],width=int(GG[u][v]["weight"]))

    plt.axis('off')
    plt.show()

def plot_assortativity_weight(G,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)
    
    weights = ut.get_weights(G,dictionary = True)
    neig_weights = ut.get_neigh_weights(G,weights)

    w = []
    nw = []

    for node in G.nodes():
        w.append(weights[node])
        nw.append(neig_weights[node])

    w = np.array(w) / np.max(w)
    nw = np.array(nw) / np.max(nw)
    plt.figure(figsize=figsize)
    plt.title("Weight Assortativity"+title)
    plt.scatter(nw,w,s=s,color=color)
    plt.xlabel("weight")
    plt.ylabel("<Knn(weight)>")

    plt.show()

def plot_assortativity_degree(G,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)
    avg_Ndegrees = list(nx.average_neighbor_degree(G).values())
    avg_Ndegrees = np.array(avg_Ndegrees)/np.max(avg_Ndegrees)

    degs = list(nx.degree(G))
    degrees = [i for i,j in degs]
    degrees = np.array(degrees)/np.max(degrees)
    plt.figure(figsize=figsize)
    plt.title("Degree Assortativity"+title)
    plt.scatter(avg_Ndegrees,degrees,s=s,color=color)
    plt.xlabel("K")
    plt.ylabel("<Knn(K)>")

    plt.show()

def plot_dist_eigvals(G,binary_adj_matrix=False,name_dataset=None,bins=40,color="blue",figsize=(5,5)):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)
    plt.figure(figsize=figsize)
    if (binary_adj_matrix):
        adj = nx.adj_matrix(G).A 
        
        for i in range(len(adj)):    # binarizze adj matrix
            for j in range(len(adj[i])):
                if not(adj[i][j] == 0):
                    adj[i][j] = 1

        G1 = nx.from_numpy_matrix(adj)        
        L = nx.normalized_laplacian_matrix(G1)   
        plt.title("Dist. eigenvalues binary adj"+title)
    else: 
        L = nx.normalized_laplacian_matrix(G)
        plt.title("Dist. eigenvalues weighted adj"+title)

    e = np.linalg.eigvals(L.A)
    plt.hist(e, bins=bins,color=color)
    plt.show() 

def plot_dist_degree(G,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    nodes_weights = []
    degrees = []
    for node in G.nodes():
        degrees.append(G.degree(node))
    
    if (normed):
        degrees = np.array(degrees)/np.max(degrees)
    plt.figure(figsize=figsize)
    plt.title("Degree"+title)
    plt.hist(degrees,bins=50,density=density,color=color)
    plt.show()

def plot_dist_weights(G,bins=40,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    nodes_weights = ut.get_weights(G)
    
    if (normed):
        nodes_weights = np.array(nodes_weights)/np.max(nodes_weights)
    plt.figure(figsize=figsize)
    plt.title("Weights"+title)
    plt.hist(nodes_weights,bins=bins,density=density,color=color)
    plt.show()
    
def plot_dist_neig_weights(G,bins=40,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    nodes_weights = ut.get_neigh_weights(G)
    
    if (normed):
        nodes_weights = np.array(nodes_weights)/np.max(nodes_weights)
    plt.figure(figsize=figsize)
    plt.title("Neighbourhood weights"+title)
    plt.hist(nodes_weights,bins=bins,density=density,color=color)
    plt.show()

def plot_dist_weights_gaps(data,gaps,density=True,name_dataset=None,bins=40,figsize=(5,5)):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    plt.figure(figsize=figsize)
    for g in gaps:
        GG = cs.build_weighted_graph(data,g)

        nodes_weights = []
        degrees = []
        for node in GG.nodes():
            degrees.append(GG.degree(node))
            links = GG.edges(node)
            node_weights = 0
            for i in links:
                node_weights = node_weights + GG[i[0]][i[1]]["weight"]

            nodes_weights.append(node_weights)

        #nodes_weights = np.array(nodes_weights)/np.max(nodes_weights)
        #degrees = np.array(degrees)/np.max(degrees)


        plt.title("Weights"+title)
        plt.hist(nodes_weights,density=density,bins=bins,label="GAP ="+str(g),alpha=0.5)

    plt.legend()
    plt.show()

def plot_dist_closeness_cent(G,bins=40,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    closensess = ut.closeness_centrality(G)
    
    if (normed):
        closensess = np.array(closensess)/np.max(closensess)
    plt.figure(figsize=figsize)
    plt.title("Closeness centrality"+title)
    plt.hist(closensess,bins=bins,density=density,color=color)
    plt.show()

# def plot_dist_degree_cent(G,bins=40,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
#     if (name_dataset == None):
#         title = ""
#     else:
#         title = "\n dataset = "+str(name_dataset)

#     degree = ut.degree_centrality(G)
    
#     if (normed):
#         degree = np.array(degree)/np.max(degree)
#     plt.figure(figsize=figsize)
#     plt.title("Degree centrality"+title)
#     plt.hist(degree,bins=bins,density=density,color=color)
#     plt.show()

def plot_dist_between_cent(G,bins=40,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    between = ut.betweenness_centrality(G)
    
    if (normed):
        between = np.array(between)/np.max(between)
    plt.figure(figsize=figsize)
    plt.title("Between centrality"+title)
    plt.hist(between,bins=bins,density=density,color=color)
    plt.show()

def plot_dist_clust_coeff(G,bins=40,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
    if (name_dataset == None):
        title = ""
    else:
        title = "\n dataset = "+str(name_dataset)

    clust_coef = ut.clustering_coeff(G)
    
    if (normed):
        clust_coef = np.array(clust_coef)/np.max(clust_coef)
    plt.figure(figsize=figsize)
    plt.title("Clustering coeff."+title)
    plt.hist(clust_coef,bins=bins,density=density,color=color)
    plt.show()


def summary(G,data,name_dataset,gaps):
    spect_binary = ut.spectral_gap(G,binary_adj_matrix=True)
    spect_weighted = ut.spectral_gap(G,binary_adj_matrix=False)

    print("spectral gap on binary adj: \t"+str(spect_binary))
    print("spectral gap on weighted adj: \t"+str(spect_weighted))
    # node_weights = ut.get_weights(G)
    # a,c,cut = ut.find_a_c_cut(node_weights,40,50,1)
    # node_neigh_weights = ut.get_neigh_weights(G)
    # aN,cN,cutN = ut.find_a_c_cut(node_neigh_weights,40,50,1)
    # print(tabulate([["Spectral GAP","weighted matrix",spect_weighted],
    #             ["Spectral GAP","binary matrix",spect_binary],
    #             ["Power low NODE_WEIGHT","a",a],
    #             ["Power low NODE_WEIGHT","c",c],
    #             ["Power low NODE_WEIGHT","cut",cut],
    #             ["Power low NODE_NEIG","a",aN],
    #             ["Power low NODE_NEIG","c",cN],
    #             ["Power low NODE_NEIG","cut",cutN]]))
    plot_assortativity_weight(G,name_dataset=name_dataset)
    plot_assortativity_degree(G,name_dataset=name_dataset)
    plot_dist_eigvals(G,False,name_dataset=name_dataset )
    plot_dist_eigvals(G,True,name_dataset=name_dataset)
    plot_dist_degree(G,name_dataset=name_dataset)
    plot_dist_weights(G,name_dataset=name_dataset)
    plot_dist_neig_weights(G,name_dataset=name_dataset)
    plot_dist_weights_gaps(data,gaps)
    plot_dist_closeness_cent(G,name_dataset=name_dataset)
    plot_dist_degree_cent(G,name_dataset=name_dataset)
    plot_dist_between_cent(G,name_dataset=name_dataset)
    plot_dist_clust_coeff(G,name_dataset=name_dataset)


 