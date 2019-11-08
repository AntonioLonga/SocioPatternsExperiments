from tabulate import tabulate
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utilities as ut
import construction as cs

def plot_assortativivity_weights_graphs(data_sets,gap=19,figsize=(18,12),s=3,color="blue",save=False):
    i = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        weights = ut.get_weights(G,dictionary = True)
        neig_weights = ut.get_neigh_weights(G,weights)

        w = []
        nw = []

        for node in G.nodes():
            w.append(weights[node])
            nw.append(neig_weights[node])

        w = np.array(w) / np.max(w)
        nw = np.array(nw) / np.max(nw)
        
        plt.subplot(2,3,i)
        plt.title("Weight Assortativity"+title)
        plt.scatter(nw,w,s=s,color=color)
        plt.xlabel("weight")
        plt.ylabel("<Knn(weight)>")

        i = i + 1
    plt.show()
    
    if (save):
        f.savefig("Weight Assorattivity.pdf")
    
    
def plot_assortativivity_degree_graphs(data_sets,gap=19,figsize=(18,12),s=3,color="blue",save=False):
    i = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        
        
        
        avg_Ndegrees = list(nx.average_neighbor_degree(G).values())
        avg_Ndegrees = np.array(avg_Ndegrees)/np.max(avg_Ndegrees)

        degs = list(nx.degree(G))
        degrees = [i for i,j in degs]
        degrees = np.array(degrees)/np.max(degrees)
        
        plt.subplot(2,3,i)
        plt.title("Degree Assortativity"+title)
        plt.scatter(avg_Ndegrees,degrees,s=s,color=color)
        plt.xlabel("K")
        plt.ylabel("<Knn(K)>")

        i = i + 1
    plt.show()
    
    if (save):
        f.savefig("Degree Assorattivity.pdf")
    
    


def plot_dist_eigvals_graphs(data_sets,binary_adj_matrix=False,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)
        
        if (binary_adj_matrix):
            adj = nx.adj_matrix(G).A 

            for i in range(len(adj)):    # binarizze adj matrix
                for j in range(len(adj[i])):
                    if not(adj[i][j] == 0):
                        adj[i][j] = 1

            G1 = nx.from_numpy_matrix(adj)        
            L = nx.normalized_laplacian_matrix(G1)   
            plt.title("Dist. eigenvalues binary adj"+title)
            name_to_save = "Eigenvalues dist. Binary ADJ.pdf"
        else: 
            L = nx.normalized_laplacian_matrix(G)
            plt.title("Dist. eigenvalues weighted adj"+title)
            name_to_save = "Eigenvalues dist. Weighted ADJ.pdf"

            
            
            
            
            
        e = np.linalg.eigvals(L.A)
        plt.hist(e, bins=bins,color=color) 

        k = k + 1
    plt.show()
    
    if (save):
        f.savefig(name_to_save)
    
def plot_dist_degree_graphs(data_sets,normed=True,density=True,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)
        
        nodes_weights = []
        degrees = []
        for node in G.nodes():
            degrees.append(G.degree(node))

        if (normed):
            degrees = np.array(degrees)/np.max(degrees)

        plt.title("Degree dist"+title)
        plt.hist(degrees,bins=bins,density=density,color=color)


        k = k + 1
    plt.show()
    
    if (save):
        f.savefig("Degree dist.pdf")
    
    

def plot_dist_weights_graphs(data_sets,normed=True,density=True,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)
        
        nodes_weights = ut.get_weights(G)
    
        if (normed):
            nodes_weights = np.array(nodes_weights)/np.max(nodes_weights)

        plt.title("Weights dist"+title)
        plt.hist(nodes_weights,bins=bins,density=density,color=color)


        k = k + 1
    plt.show()
    
    if (save):
        f.savefig("Weights dist.pdf")
    
    
def plot_dist_neigh_weights_graphs(data_sets,normed=True,density=True,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)
        
        nodes_weights = ut.get_neigh_weights(G)
    
        if (normed):
            nodes_weights = np.array(nodes_weights)/np.max(nodes_weights)

        plt.title("Weights neigh dist"+title)
        plt.hist(nodes_weights,bins=bins,density=density,color=color)


        k = k + 1
    plt.show()
    
    if (save):
        f.savefig("Weights neigh dist.pdf")
    


def plot_dist_closeness_cent_graphs(data_sets,normed=True,density=True,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)
        closensess = ut.closeness_centrality(G)

        if (normed):
            closensess = np.array(closensess)/np.max(closensess)
            
        plt.title("Closeness centrality dist"+title)
        plt.hist(closensess,bins=bins,density=density,color=color)


        k = k + 1
    plt.show()
    
    if (save):
        f.savefig("Clos cent dist.pdf")
    
    

def plot_dist_between_cent_graphs(data_sets,normed=True,density=True,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)

        
        between = ut.betweenness_centrality(G)

        if (normed):
            between = np.array(between)/np.max(between)
        plt.title("Between centrality"+title)
        plt.hist(between,bins=bins,density=density,color=color)

        k = k + 1
    plt.show()
    
    if (save):
        f.savefig("Between cent dist.pdf")
    


   
   
def plot_dist_clust_coeff_graphs(data_sets,normed=True,density=True,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False):
    k = 1
    f = plt.figure(figsize=figsize)
    
    names_dataset = list(data_sets.keys())
    datas = list(data_sets.values())
    
    for j in range(len(datas)):
        
        title = "\n dataset = "+str(names_dataset[j])
        G = cs.build_weighted_graph(datas[j],gap)
        plt.subplot(2,3,k)

        clust_coef = ut.clustering_coeff(G)

        if (normed):
            clust_coef = np.array(clust_coef)/np.max(clust_coef)
        plt.title("Clustering coeff."+title)
        plt.hist(clust_coef,bins=bins,density=density,color=color)



        k = k + 1
    plt.show()
    
    if (save):
        f.savefig("Clustering coeff dist.pdf")



    
def summary_graphs(data_sets,gap=19,save=False):

    
    names = list(data_sets.keys())
    datas = list(data_sets.values())

    del names[1]
    del datas[1]
    del names[4]
    del datas[4]
    s_b,s_w,a,c,cut,aN,cN,cutN = summary(datas,gap)
    
    print(tabulate([["Spectral GAP","w matrix",s_w[0],s_w[1],s_w[2],s_w[3]],
                    ["Spectral GAP","b matrix",s_b[0],s_b[1],s_b[2],s_b[3]],
                    ["Power low NODE_WEIGHT","a",a[0],a[1],a[2],a[3]],
                    ["Power low NODE_WEIGHT","c",c[0],c[1],c[2],c[3]],
                    ["Power low NODE_WEIGHT","cut",cut[0],cut[1],cut[2],cut[3]],
                    ["Power low NODE_NEIG","a",aN[0],aN[1],aN[2],aN[3]],
                    ["Power low NODE_NEIG","c",cN[0],cN[1],cN[2],cN[3]],
                    ["Power low NODE_NEIG","cut",cutN[0],cutN[1],cutN[2],cutN[3]]],
                   headers=['','',names[0],names[1],names[2],names[3]]))


    if save:
        f = open('table.txt', 'w')
        f.write(tabulate([["Spectral GAP","w matrix",s_w[0],s_w[1],s_w[2],s_w[3]],
                    ["Spectral GAP","b matrix",s_b[0],s_b[1],s_b[2],s_b[3]],
                    ["Power low NODE_WEIGHT","a",a[0],a[1],a[2],a[3]],
                    ["Power low NODE_WEIGHT","c",c[0],c[1],c[2],c[3]],
                    ["Power low NODE_WEIGHT","cut",cut[0],cut[1],cut[2],cut[3]],
                    ["Power low NODE_NEIG","a",aN[0],aN[1],aN[2],aN[3]],
                    ["Power low NODE_NEIG","c",cN[0],cN[1],cN[2],cN[3]],
                    ["Power low NODE_NEIG","cut",cutN[0],cutN[1],cutN[2],cutN[3]]],
                   headers=['','',names[0],names[1],names[2],names[3]]))
        f.close()
  



def summary(data,gap):
    spec_b = []
    spec_w = []
    a = []
    c = []
    cut = []
    aN = []
    cN = []
    cutN = []
    log = 1
    for d in data:
        print(log)
        log = log + 1 
        G = cs.build_weighted_graph(d,gap)

        spec_b.append("{:.4f}".format(ut.spectral_gap(G,binary_adj_matrix=True)))
        spec_w.append("{:.4f}".format(ut.spectral_gap(G,binary_adj_matrix=False)))

        node_weights = ut.get_weights(G)
        aa,cc,ccut = ut.find_a_c_cut(node_weights,40,150,1)
        a.append("{:.4f}".format(aa))
        c.append("{:.4f}".format(cc))
        cut.append(int(ccut))
        node_neigh_weights = ut.get_neigh_weights(G)
        aaN,ccN,ccutN = ut.find_a_c_cut(node_neigh_weights,40,150,1)
        aN.append("{:.4f}".format(aaN))
        cN.append("{:.4f}".format(ccN))
        cutN.append(int(ccutN))
        
        
    return(spec_b,spec_w,a,c,cut,aN,cN,cutN)