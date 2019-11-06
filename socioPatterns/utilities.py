import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_weights(G,dictionary = False):
    weights = dict()
    for node in G.nodes():
        weight = 0
        for e in G.edges(node):
            weight = weight + G[e[0]][e[1]]["weight"]
        weights[node] = weight
    if (dictionary == False):
        weights = np.array(list(weights.values()))
    return(weights)




def get_neigh_weights(G,dictionary = False):
    w = get_weights(G,True)
    neig_w = dict()
    for node in G.nodes():
        neig = []
        for e in G.edges(node):
            neig.append(w[e[1]])
        neig_w[node] = np.mean(neig)
        
    if (dictionary == False):
        neig_w = np.array(list(neig_w.values()))
    return(neig_w)




def clustering_coeff(G,dictionary=False):
    clustering = nx.clustering(G)
    if not (dictionary):
        clustering = np.array(list(clustering.values()))
    
    return(clustering)



def betweenness_centrality(G,dictionary=False):
    betweenness_centrality = nx.betweenness_centrality(G)
    if not (dictionary):
        betweenness_centrality = np.array(list(betweenness_centrality.values()))
    
    return(betweenness_centrality)



def closeness_centrality(G,dictionary=False):
    closeness_centrality = nx.closeness_centrality(G)
    if not (dictionary):
        closeness_centrality = np.array(list(closeness_centrality.values()))
    
    return(closeness_centrality)


def degree_centrality(G,dictionary=False):
    degree_centrality = nx.degree_centrality(G)
    if not (dictionary):
        degree_centrality = np.array(list(degree_centrality.values()))
    
    return(degree_centrality)