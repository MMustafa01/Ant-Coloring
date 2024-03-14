import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class AntCol:
    def __init__(self,
                 seed               : int   = 0,
                 alpha              : float = 0.8,
                 beta               : float = 0.8,
                 Q                  : float = 4,
                 Numant             : int   = 20,
                 rho                : float = 0.8, # evaporation rate
                 dataset            : str   = 'queen11_11.col',
                 iterations         : int   = 10
                  ):
        self.rng = np.random.default_rng(seed)
        self.ant_num = Numant
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.rho = rho
        self.graph_nodes_num = None
        self.dataset = dataset
        self.iterations = iterations
        self.data_loader()


        
        
        
    def data_loader(self):
        
        with open(self.dataset, 'r') as fhandle:
            lst = [i.strip().split() for i in fhandle]
            self.graph_nodes_num = int(lst[0][2])
            num_edges = int(lst[0][3])
            self.graph = np.zeros((self.graph_nodes_num,self.graph_nodes_num))
            for i in range(1, num_edges + 1):
                n1 = int(lst[i][1])
                n2 = int(lst[i][2])
                self.graph[n1-1, n2-1] = self.graph[n2-1, n1-1 ] =  1# self.get_distance(n1,n2)
    def ant_col(self):
        pharamone_trail = self.init_pharamones()
        # initialize best solutions with inf and a null set
        best_num_color = np.inf
        best_so_far = []
        average_so_far = []
        
        for it in tqdm(range(self.iterations)):
            delta_pharamones = np.zeros(shape=pharamone_trail.shape)
            q_for_every_ant=[]
            for  i in tqdm(range(self.ant_num)):
                # Get a a solution in the form s = (V1, V2,V3....VQ) 
                solution,q  = self.construct_graph(pharamone_trail)
                q_for_every_ant.append(q)
                if  q < best_num_color:
                    best_num_color = q
                    best_solution = solution
                # Delta_pharamones should be 0 where ever there is an edge
                delta_pharamones[delta_pharamones != 0] += self.Q/len(solution)
            x = (1- self.rho)* pharamone_trail + delta_pharamones
            pharamone_trail +=  (1- self.rho)* pharamone_trail + delta_pharamones
            print(f"Pharamone trail =\n{x.shape}")
            average_so_far.append(np.average(q_for_every_ant))

            best_so_far.append(np.min(q_for_every_ant))

        return best_solution,best_num_color, best_so_far,average_so_far
    def main(self)-> None:
        best_solution,best_num_color, best_so_far,average_so_far =self.ant_col()
        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot best-so-far values
        plt.plot( best_so_far, label='Best So Far')

        # Plot average values
        plt.plot( average_so_far, label='Average so far')

        # Customize the plot
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.title('Best So Far and Average Values')
        plt.grid(True)
        plt.legend()

        # Display the plot
        plt.show()

        return None 

    def construct_graph(self, pharamone_trail):
        '''
        Using the DSATUR heuristic
        '''
        
     
        d_sat = np.array([0]*self.graph_nodes_num)

        c_min = np.array([0]*self.graph_nodes_num) #colour index starts with zero
        #Using set because removing elements is easier and because its cool xD
        A = set(np.arange(self.graph_nodes_num))  #Unvisited set
        q = 1 # Number of colors used
        degrees = np.array([self.degree(vertex)[0] for vertex in range(self.graph_nodes_num)])
        self.max_coloring = np.max(degrees) 
        current_node = np.argmax(degrees)
        # The solution is a deictionary where the weights are a index corresponding to some color. The length of the dictornary is equal to the upper bound of the chromatic number
        partial_solution  = [set() for color in range(self.max_coloring)] #index corresponds to the color
        partial_solution[0].add(current_node)
        for i in range(1, self.graph_nodes_num):
            A.remove(current_node)
            neighbors = self.get_neighbours(current_node)
            # self.update_cmin_N(c_min, neighbors)
            self.update_dsat_N_c_min(d_sat,c_min, neighbors, partial_solution)
            
            # Implementing stratergy 1
        
            vertices, probabilities = self.get_probability(partial_solution, A,c_min, pharamone_trail,d_sat)
            current_node = self.stochastic_sampling(vertices, probabilities) 
            color = c_min[current_node]

            partial_solution[color].add(current_node)
            if color >= q + 1:
                q = color +1
        solution = partial_solution.copy()
        return solution,q


    def update_dsat_N_c_min(self, dsat, c_min, neighbors, partial_solution):
        for neighbor in neighbors:
            unique_color = set()
            for neb_squared in self.get_neighbours(neighbor):
                # dsat is the number of different colours already assigned to vertex adjacent to A
                color = self.get_color(neb_squared, partial_solution)
                if color != -1:
                    unique_color.add(color)
            dsat[neighbor] = len(unique_color)
            for color in range(self.max_coloring):
                if color not in unique_color:
                    c_min[neighbor] = color
                    break
        # print("/////")
        # print(f"Unique color {unique_color}")
        # print(f"cmin {c_min[neighbor]}")
        # print(f'd_sat {dsat[neighbor]}')
        # print("////")
      
    def get_color(self, v, solution):
        
        for color in range(len(solution)): # color is the index of the set in the solution list
            if v in solution[color]: #
                return color
        return  -1 # returned when vertex is not colored

 
    def trail_factor(self, vertex, solution,color, pharamone_trail )->float:
        return 1 if not solution[color] else np.sum([pharamone_trail[x, vertex] for x in solution[color]])/len(solution[color])
        
    def get_probability(self,solution, A,c_min, pharamone_trail,d_sat ):
        probabilities = []

        vertices = []
        for v in A:
            trailFactor = self.trail_factor(v, solution, c_min[v], pharamone_trail)
            desirability = d_sat[v]
            denominator = [self.trail_factor(vertex, solution, c_min[vertex], pharamone_trail)**self.alpha*d_sat[vertex]**self.beta for vertex in A] 
            
            probability = (trailFactor**self.alpha*desirability**self.beta)/np.sum(denominator)
            probabilities.append( probability)
  
            vertices.append(v)
            

        return vertices,probabilities


    def stochastic_sampling(self,items, probabilities):
        """
        Selects a random item from a list based on the corresponding probabilities.

        Args:
            items: A list of items.
            probabilities: A list of probabilities corresponding to each item in 'items'.

        Returns:
            A randomly selected item from the list.
        """
        #tqdm.write(f'{probabilities}')
        # Validate input lengths
        if len(items) != len(probabilities):
            raise ValueError("Length of items and probabilities must be equal.")

        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(probabilities)
        if total_prob != 1:
            probabilities = [prob / total_prob for prob in probabilities]

        # Create a cumulative distribution function (CDF)
        cdf = [0] * len(probabilities)
        cdf[0] = probabilities[0]
        for i in range(1, len(probabilities)):
            cdf[i] = cdf[i-1] + probabilities[i]

        # Generate a random number between 0 and 1
        random_value = np.random.random()

        # Find the item corresponding to the random value using the CDF
        #tqdm.write(f'cdf[-1] = {cdf[-1]}')
        for i, prob in enumerate(cdf):
            if random_value <= prob:
                return items[i]

            



 

    def init_pharamones(self) -> np.ndarray:
        pharamone_trail = np.ones(self.graph.shape)
        # print(self.pharamone_trail.shape)
        for i in range(self.graph.shape[0]):
            neighbors = self.get_neighbours(i)
            pharamone_trail[i,neighbors] = 0
        return pharamone_trail

  
        
    def get_neighbours(self, v) :
        neighbors = list()
        for i in range(len(self.graph[v])):
            if not self.graph[v,i] == 0:
                neighbors.append(i)
        return neighbors
    

    def degree(self,v) -> int:
        neighbors = self.get_neighbours(v)
        return len(neighbors),neighbors

    
 
    def is_feasible(self) ->bool:
        return True
    
if __name__ == '__main__':
    alpha = 0.5
    beta = 0.5
    gamma = 0.5
    iterations = 10
    num_ants = 20
    dataset = 'queen11_11.col'

    graph_coloring = AntCol(alpha= alpha,
                            beta=beta,
                            rho= gamma,
                            iterations=iterations,
                            num_ants = num_ants,
                            dataset= dataset)
    graph_coloring.main()
