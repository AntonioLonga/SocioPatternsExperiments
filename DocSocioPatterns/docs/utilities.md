# Utility

Import the library using 
```
import utilities
```


## get_weights
```
utilities.get_weights(graph, dictionary = False)
```
### Input
* graph = A networkx graph.
* dictionary = If it is true then returns a dictionary [node_id : weights].  
If it is false then returns an np.array of weights.
### Output
* weights = np.ndarray or dictionary of weights.



## get_neigh_weights
```
utilities.get_neigh_weights(Graph, dictionary = False)
```
### Input
* graph = A networkx graph.
* dictionary = If it is true then returns a dictionary [node_id : weights]. 
If it is false then returns an np.array.
### Output
* weights = np.ndarray or dictionary of weights, where each weigth is the sum of the weights of the neighbours of the node.



## clustering_coeff
```
utilities.clustering_coeff(Graph, dictionary = False)
```
### Input
* graph = A networkx graph.
* dictionary = If it is true then returns a dictionary [node_id : weights]. 
If it is false then returns an np.array.
### Output
* coeff = np.ndarray or dictionary of coefficients.



## betweenness_centrality
```
utilities.betweenness_centrality(Graph, dictionary = False)
```
### Input
* graph = A networkx graph.
* dictionary = If it is true then returns a dictionary [node_id : weights]. 
If it is false then returns an np.array.
### Output
* coeff = np.ndarray or dictionary of values.



## betweenness_centrality
```
utilities.betweenness_centrality(Graph, dictionary = False)
```
### Input
* graph = A networkx graph.
* dictionary = If it is true then returns a dictionary [node_id : weights]. 
If it is false then returns an np.array.
### Output
* coeff = np.ndarray or dictionary of values.






## spectral_gap
```
utilities.spectral_gap(Graph, binary_adj_matrix=False)
```
### Input
* graph = A networkx graph.
* binary_adj_matrix = If it is true then returns the spectral gap using an unweighted adj matrix, while if it is false then returns the spectral gap computed using an adj matrix where the element in position i, j is equal to the weights on edge from node i to node j.
### Output
* spectral_gap = It returns a float representing the gap between the first and the second smallest eigenvalue.






## find_a_c_cut
```
utilities.find_a_c_cut(data,min_bin=40,max_bin=150,interval=1)
```
### Input
* data = a file where you want to fit a power low.
* min_bin = smallest number of bins
* max_bin = largest number of bins
* intervals = np.arange(min_bin,max_bin,intervals)
### Output
* a = exponent of the fitted power low.
* c = constant of the fitted power low.
* cut = position\ of where the power low is fitted.