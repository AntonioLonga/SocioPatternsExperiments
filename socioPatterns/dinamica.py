import numpy as np


class simulator_dynamic():
    def __init__(self,events,probability = 0.5,initial_node = None, seed=None):
        self.events = events
        self.reached_nodes = []
        self.individals = list(events[0].nodes())

        if not (seed == None):
            print("in")
            np.random.seed(seed)


        self.set_inital_state(initial_node)
        self.probability = probability

        self.results = []

        
       
    def set_inital_state(self,node = None):
        if (node == None):
            np.random.shuffle(self.individals)
            node = self.individals[0]

        # verify that the node is in the graph
        assert (self.events[0].has_node(node)),"DINAMICA: The initial node is not in the graph"

        self.inital_state = node # set the node
        self.reached_nodes = [node] # reset reached nodes, and add initial state


    def run(self):
        for i in range(len(self.events)): # per tutti i grafi
            tmp = []
            for j in self.reached_nodes: # per tutti i nodi con l info
                if (self.events[i].has_node(j)):
                    e = self.events[i].edges(j)  
                    for k,h in e: # per tutti i nodi collegati al nodo j
                        random = np.random.rand()
                        if (random <= self.probability): 
                            tmp.append(h) # aggiungi il nodo a tmp

                self.reached_nodes = list(self.reached_nodes) + tmp    
                self.reached_nodes = list(np.unique(self.reached_nodes)) # unisci tmp a reached nodes

            self.results.append(self.reached_nodes)


def get_percentages_of_consent(simulator_dynamic):

    individals = simulator_dynamic.individals
    percentages = []

    for i in simulator_dynamic.results:
        percentages.append(len(i)/len(individals))
        
    return(percentages)
    

def get_frequence_of_consent(simulator_dynamic):

    individals = simulator_dynamic.individals
    frequence = {key: 0 for key in individals}

    
    for i in simulator_dynamic.results:
        for j in i:
            frequence[j] = frequence[j] + 1 

    return frequence


def simulate_n_times(times,graphs,probability,initial_node = None,seed=None):
    # ritorna una array di array e una rray di dict
    res_precentage = []
    res_frequences = []
    for i in range(times):
        sim_dyn = simulator_dynamic(graphs,probability,initial_node,seed)
        sim_dyn.run()
        res_precentage.append(get_percentages_of_consent(sim_dyn))
        res_frequences.append(get_frequence_of_consent(sim_dyn))

    return (res_precentage,res_frequences)