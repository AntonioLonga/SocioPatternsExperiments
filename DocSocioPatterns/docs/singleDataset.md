# Analysis on single data set

Can be used to plot several properties of a data set.

```
import singleDatasetAnalysis as san
```


## plot_weighted_graph
```
san.plot_weighted_graph(graph, pos, node_size=70)
```
### Input
* graph = A weighted networkx graph.
* pos = networkx position.
* node_size = size of the plotted nodes.
### Output
* Plot the graph where the width of the edge represent the number of interactions.


## plot_assortativity_weight
```
san.plot_assortativity_weight(graph,name_dataset=None,color="blue",figsize=(5,5),s=3)
```
### Input
* graph = A weighted networkx graph.
### Output
* Plot the assortativity concerning the weights.


## plot_assortativity_degree
```
san.plot_assortativity_degree(graph,name_dataset=None,color="blue",figsize=(5,5),s=3)
```
### Input
* graph = A networkx graph.
### Output
* Plot the assortativity concerning the degree.




## plot_dist_eigvals
```
san.plot_dist_eigvals(graph,binary_adj_matrix=False,name_dataset=None,bins=40,color="blue",figsize=(5,5))
```
### Input
* graph = A networkx graph.
* binary_adj_matrix = if it is false then it uses the weighted adj matrix, while if it is true then it uses the binary adj matrix.
### Output
* Plot the histogram of the eigenvalues.




## plot_dist_degree
```
san.plot_dist_degree(graph,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* graph = A networkx graph.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the degrees.




## plot_dist_weights
```
san.plot_dist_weights(graph,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* graph = A networkx graph.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the weights.




## plot_dist_neig_weights
```
san.plot_dist_neig_weights(graph,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* graph = A networkx graph.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the weights of the neighbours of a node.



## plot_dist_weights_gaps
```
san.plot_dist_weights_gaps(data,gaps,density=True,name_dataset=None,bins=40,figsize=(5,5))
```
### Input
* data = the input file.
* gaps = a list of different gaps that are tested.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)

### Output
* Plot the histogram of the weights for different gaps using different colors.




## plot_dist_closeness_cent
```
san.plot_dist_closeness_cent(graph,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* graph = A networkx graph.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the closeness centrality of the nodes.





## plot_dist_between_cent
```
san.plot_dist_between_cent(graph,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* graph = A networkx graph.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the betweeness centrality of the nodes.







## plot_dist_clust_coeff
```
san.plot_dist_between_cent(graph,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* graph = A networkx graph.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the clustering coefficients of the nodes.





## summary
```
san.summary(graphs,data,name_dataset,gaps)
```
### Input
* graph = A networkx graph.
* data = the input data.
* gaps = a list of different gaps that are tested.
### Output
* Plot all the previous plots and prints spectral gap.


