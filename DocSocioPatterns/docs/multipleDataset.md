# Analysis on multiple data sets

Can be used to plot several properties of a set of data sets.

```
import multipleDatasetAnalysis as man
```



## plot_assortativivity_weights_graphs
```
man.plot_assortativity_weight(data_sets,gap=19,name_dataset=None,color="blue",figsize=(5,5),s=3)
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
### Output
* Plot the assortativity concerning the weights for each data set.



## plot_assortativivity_degree_graphs
```
man.plot_assortativity_degree(data_sets,gap=19,name_dataset=None,color="blue",figsize=(5,5),s=3)
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
### Output
* Plot the assortativity concerning the degrees for each data set.



## plot_dist_eigvals_graphs
```
man.plot_dist_eigvals_graphs(data_sets,binary_adj_matrix=False,gap=19,bins=40,figsize=(18,12),s=3,color="blue",save=False)
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* binary_adj_matrix = if it is false then it uses the weighted adj matrix, while if it is true then it uses the binary adj matrix.
### Output
* Plot the histogram of the eigenvalues for each data set.




## plot_dist_degree_graphs
```
man.plot_dist_degree_graphs(data_sets,normed=True,density=True,gap=19,bins=40,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the degrees for each data set.




## plot_dist_weights_graphs
```
man.plot_dist_weights_graphs(data_sets,normed=True,density=True,gap=19,bins=40,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the weights for each data set.




## plot_dist_neigh_weights_graphs
```
man.plot_dist_neigh_weights_graphs(data_sets,normed=True,density=True,gap=19,bins=40,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the weights of the neighbours of a node for each data set.



## plot_dist_closeness_cent_graphs
```
man.plot_dist_closeness_cent_graphs(data_sets,gap=19,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the closeness centrality of the nodes for each data set.





## plot_dist_between_cent_graphs
```
man.plot_dist_between_cent_graphs(data_sets,gap=19,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the betweeness centrality of the nodes for each data set.





## plot_dist_clust_coeff_graphs
```
man.plot_dist_clust_coeff_graphs(data_sets,gap=19,normed=True,density=True,name_dataset=None,color="blue",figsize=(5,5),s=3):
```
### Input
* data_sets = An array of loaded data sets.
* gap = gap of time used to split the data sets.
* normed = If it is true, the values are normalized.
* density = plt.hist(..., densty, ...)
### Output
* Plot the histogram of the clustering coefficients of the nodes for each data set.



