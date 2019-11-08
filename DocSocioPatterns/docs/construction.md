# Construction

Import the library using 
```
import construction
```


## split_data_in_groups
```
construction.split_data_in_groups(path_data,path_metadata)
```
### Input
* path_data = path of the input data.
* path_metadata = path of the input metadata
### Output
* gropus = a dictionary \[group : data\], where group is the name taken from the metadata, while data are the data related to a sepcific group.



## load_data
```
construction.load_data(path)
```
### Input
* path = path of the input data.
### Output
* data = np.ndarray of the loaded data. Each row t,i,j represent the interaction among node i and node j at time t.



## individuals
```
construction.individuals(path)
```
### Input
* data = The loaded file.
### Output
* individuals = np.ndarray of individuals in the dataset.






## build_weighted_graph
```
construction.build_weighted_graph(data,gap=19)
```
### Input
* data = The loaded file.
* gap = A gap of time.
### Output
* Graph = It returns a weighted nx.Graph, where the weight on a edge i-j represent the number of interactions among node i and node j.





## build_weighted_graph_2
```
construction.build_weighted_graph_2(data,gap=19)
```
### Input
* data = The loaded file.
* gap = A gap of time.
### Output
* Graph = It returns a weighted nx.Graph, where the weight on a edge i-j represent the number of interactions among node i and node j. With this construction, consecutive interaction are counted as a single interaction.





## build_graphs
```
construction.build_graphs(data,gap=19)
```
### Input
* data = The loaded file.
* gap = A gap of time.
### Output
* array of Graphs = It returns an array of graphs, each graph represent the interaction of nodes, within time t and time t+gap.







## split_input_data
```
construction.split_input_data(data, gap=19)
```
### Input
* data = The loaded file.
* gap = A gap of time.
### Output
* np.array = an array of data, each element represent the interaction within the gap of time. 

