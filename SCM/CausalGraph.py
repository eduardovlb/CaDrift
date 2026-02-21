import numpy as np
import random
from SCM.Mappers import *
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import json

from collections import Counter

import copy

import warnings
warnings.filterwarnings("ignore")

class Vertex:
    """A vertex/node in a causal graph representing a variable."""
    def __init__(self, name: str, value: float = None, mapper : Mapping = None):
        self.name = name
        self.value = value
        self.parents = []
        self.children = []
        self.mapper : Mapping = mapper
        self.drift_history = []
        self.is_incremental_drifting = False
        self.drift_mean = 0
        self.alpha = 0.2
        self.last_value = 0
        
    def store_mapper(self):
        """Save the current mapper to drift history before overwriting it."""
        self.drift_history.append(copy.deepcopy(self.mapper))

    def get_children(self) -> list["Vertex"]:
        """Return the list of child vertices."""
        return self.children

    def update_mean(self):
        """Update the drift mean for incremental drift."""
        if self.is_incremental_drifting:
            self.drift_mean += self.mapper.drift_direction * self.mapper.drift_rate

    def add_parent(self, parent: "Vertex") -> None:
        """Add a parent vertex."""
        if parent not in self.parents:
            self.parents.append(parent)

    def add_child(self, child: "Vertex") -> None:
        """Add a child vertex."""
        if child not in self.children:
            self.children.append(child)

    def remove_parent(self, parent: "Vertex") -> None:
        """Remove a parent vertex."""
        if parent in self.parents:
            self.parents.remove(parent)
    
    def remove_child(self, child: "Vertex") -> None:
        """Remove a child vertex."""
        if child in self.children:
            self.children.remove(child)
        
    def get_parents(self) -> list["Vertex"]:
        """Return the list of parent vertices."""
        return self.parents

    def is_root(self) -> bool:
        """Check if the vertex is a root (no parents)."""
        return len(self.parents) == 0

    def is_leaf(self) -> bool:
        """Check if the vertex is a leaf (no children)."""
        return len(self.children) == 0

    def __str__(self):
        return f"Vertex(name={self.name}, value={self.value}, parents={self.parents}, children={self.children}, mapper={self.mapper.__str__()})"
    
    def drift_label_function(self):
        """Drift the label function mapper."""
        self.store_mapper()
        self.mapper.drift()
    
    def compute_value(self) -> None:
        """Compute the value of the vertex based on its parents and mapper."""
        if self.mapper and self.parents:
            parent_values = np.array([p.value for p in self.parents]).reshape(1, -1)
            
            if(not self.mapper.is_fitted()):    
                self.mapper.fit(len(self.parents))
            if self.is_incremental_drifting:
                self.update_mean()
                drifted_value = self.mapper.map(parent_values) + np.random.normal(loc=self.drift_mean, scale=0.1)
                self.value = float(drifted_value.ravel()[0]) if isinstance(drifted_value, np.ndarray) else float(drifted_value)
                if not np.isfinite(self.value):
                    print(f"Warning: Non-finite value encountered in incremental drift for vertex {self.name}: {self.value}. Resetting to 0.")
                    self.value = 0.0
                self.value = np.clip(self.value, -100, 100)
            else:    
                result = self.mapper.map(parent_values)
                self.value = float(result.ravel()[0]) if isinstance(result, np.ndarray) else float(result)
                if not np.isfinite(self.value):
                    print(f"Warning: Non-finite value encountered for vertex {self.name}: {self.value}. Resetting to 0.")
                    self.value = 0.0
                self.value = np.clip(self.value, -100, 100)
            
    def compute_value_last_concept(self, concept_index: int = -1) -> None:
        """Compute the value of the vertex using the mapper from a past concept."""
        if len(self.drift_history) != 0 and self.parents:
            parent_values = np.array([p.value for p in self.parents]).reshape(1, -1)
            past_mapper = self.drift_history[concept_index]

            result = past_mapper.map(parent_values)
            self.value = float(result.ravel()[0]) if isinstance(result, np.ndarray) else float(result)
            self.value = np.clip(self.value, -100, 100)

class CausalGraph:
    """
    Represents a Causal Graph using vertices and edges.
    Attributes:
    - vertices: A dictionary mapping vertex names to Vertex objects.
    - edges: A dictionary mapping vertex names to lists of child vertex names.
    - drifted_nodes: A list of nodes that have undergone drift.
    - concept_history: A list of past concepts for recurrent drift.
    - is_trained: A boolean indicating if the graph has been trained.
    Methods:
    - add_vertex(vertex): Adds a vertex to the graph.
    - add_edge(vertex1, vertex2): Adds a directed edge from vertex1 to vertex2.
    - topological_sort(): Returns a topological ordering of the vertices.
    - train_graph(train_size): Trains the graph by generating training data.
    - real_drift(incremental): Applies real drift to selected nodes.
    - virtual_drift(): Applies virtual drift to selected root nodes.
    - local_drift(): Applies local drift to a single root node.
    - severe_drift(): Applies severe drift to the label node.
    - recurrent_drift(): Reverts selected nodes to past concepts.
    - simulate(): Computes values for all vertices in topological order.
    - generate_with_unbalance(...): Generates samples with class imbalance and drift.
    - generate(...): Generates samples from the causal graph with drift and interventions.
    - steer_to_class(target_class, vtx): Steers a vertex to a target class value.
    - _generate_one_sample(sorted_vertices, drift_info): Helper to generate a single sample.    
    """
    def __init__(self, seed=42):
        self.vertices = {}
        self.edges = {}
        self.drifted_nodes = []
        self.concept_history = []
        self.is_trained = False
        self.drift_events = []  # Track all drift events
        self.current_sample_index = 0  # Track current sample index
        np.random.seed(seed)

    def add_vertex(self, vertex: Vertex) -> None:
        """Adds a vertex/node to the graph."""
        self.vertices[vertex.name] = vertex

    def add_edge(self, vertex1: Vertex, vertex2: Vertex) -> None:
        """Adds a directed edge from vertex1 to vertex2."""
        vertex1.add_child(vertex2)
        vertex2.add_parent(vertex1)

    # def topological_sort(self) -> list:
    #     """Returns a topological ordering of the vertices."""
    #     visited = set()
    #     order = []

    #     def visit(v):
    #         print(v)
    #         if v in visited:
    #             return
    #         for parent in self.vertices[v].parents:
    #             visit(parent.name)
    #         visited.add(v)
    #         order.append(v)

    #     for v in self.vertices:
    #         visit(v)
        
    #     return order

    def topological_sort(self) -> list:
        visited = set()
        visiting = set()
        order = []


        def visit(v, stack=None):
            if stack is None:
                stack = []

            if v in visiting:
                print("CYCLE:", " -> ".join(stack + [v]))
                raise RuntimeError(f"Cycle detected at {v}")

            if v in visited:
                return

            visiting.add(v)
            stack.append(v)

            for parent in self.vertices[v].parents:
                visit(parent.name, stack)

            stack.pop()
            visiting.remove(v)
            visited.add(v)
            order.append(v)


        for v in self.vertices:
            visit(v)

        return order

    
    def train_graph(self, train_size: int= 100) -> None:
        """Trains the graph by generating training data for each non-root node."""
        sorted_vertices = self.topological_sort()
        
        training_sets = {v.name: {"X": [], "y": []} for v in self.vertices.values() if not v.is_root()}
        
        for _ in range(train_size):
            
            for v in self.vertices.values():
                v.value = None

            for v_name in sorted_vertices:
                vtx: Vertex = self.vertices[v_name]

                if vtx.is_root():
                    vtx.value = float(np.asarray(vtx.mapper.map(None)).ravel()[0])
                else:                    
                    parent_values = np.array([p.value for p in vtx.parents])
                    y = vtx.mapper.generate_untrained_example(X=parent_values)
                    vtx.value = float(np.asarray(y).ravel()[0])

                    training_sets[vtx.name]["X"].append(parent_values)
                    training_sets[vtx.name]["y"].append(float(np.asarray(y).ravel()[0]))
                # if vtx.is_root():
                #     vtx.value = float(vtx.mapper.map(None))
                # else:                    
                #     parent_values = np.array([np.asarray(p.value) for p in vtx.parents])
                #     y = vtx.mapper.generate_untrained_example(X=parent_values)
                #     vtx.value = y

                #     training_sets[vtx.name]["X"].append(parent_values)
                #     training_sets[vtx.name]["y"].append(y)

        for v_name, data in training_sets.items():
            vtx = self.vertices[v_name]
            X = np.array(data["X"])
            y = np.array(data["y"])
            vtx.mapper.fit(X, y)
            vtx.training_mean = np.mean(X)

        self.is_trained = True
            
    def endogenous_drift(self, incremental: bool = False) -> None:
        """Applies endogenous drift to selected nodes. If incremental is True, applies incremental drift."""
        sorted_vertices = self.topological_sort()

        self.drifted_nodes = []

        # num_nodes_drifted = random.randint(1, 3)
        num_nodes_drifted = 1

        candidate_nodes = [v for v in self.vertices.values() if not v.is_root() and not v.name == 'y']

        if incremental:
            candidate_nodes = [v for v in candidate_nodes if isinstance(v.mapper, IncrementalMapping) or isinstance(v.mapper, AbstractCategoricalMapper)]

        if len(candidate_nodes) == 0:
            print("No eligible nodes for drift. Skipping this drift event.")
            return
        
        # label_node = self.vertices.get('y', None)
        # if label_node not in candidate_nodes:
        #     candidate_nodes.append(label_node)

        selected_nodes = []
        concept_snapshot = {}

        for _ in range(num_nodes_drifted):
            if len(candidate_nodes) == 0:
                break
            drifted_node: Vertex = np.random.choice(candidate_nodes)
            candidate_nodes.remove(drifted_node)

            drifted_node.drift_history.append(copy.deepcopy(drifted_node.mapper)) 

            training_set = {"X": [], "y": []}
            for _ in range(100):
                for v in self.vertices.values():
                    v.value = None

                for v_name in sorted_vertices:
                    vtx = self.vertices[v_name]

                    if vtx.is_root():
                        vtx.value = float(vtx.mapper.map(None))
                    else:
                        vtx.compute_value()

                    if vtx.name == drifted_node.name:
                        break

                parent_values = np.array([p.value for p in drifted_node.parents])
                y = drifted_node.mapper.generate_untrained_example(X=parent_values)

                training_set["X"].append(parent_values)
                training_set["y"].append(float(np.asarray(y).ravel()[0]))

            if not incremental:
                # abrupt drift
                drifted_node.mapper.drift(np.array(training_set["X"]), np.array(training_set["y"]))
            else:
                drifted_node.mapper.start_incremental_drift()

            selected_nodes.append(drifted_node)
            self.drifted_nodes.append(drifted_node)
                

        if not hasattr(self, 'concept_history'):
            self.concept_history = []

        concept_snapshot = {
            "nodes": selected_nodes,
            "mappers": {node.name: copy.deepcopy(node.drift_history[-1]) for node in selected_nodes}
        }
        self.concept_history.append(concept_snapshot)

    def exogenous_drift(self) -> None:
        """Applies exogenous drift to selected root nodes."""
        # num_nodes_drifted = random.randint(1,3)
        num_nodes_drifted = 1
        candidate_nodes = [v for v in self.vertices.values() if v.is_root()]

        selected_nodes = []

        self.drifted_nodes = []

        for _ in range(num_nodes_drifted):
            drifted_node : Vertex = np.random.choice(candidate_nodes)
            selected_nodes.append(drifted_node)
            drifted_node.drift_history.append(copy.deepcopy(drifted_node.mapper)) 
            drifted_node.mapper.drift()
            self.drifted_nodes.append(drifted_node)

        if not hasattr(self, 'concept_history'):
            self.concept_history = []

        concept_snapshot = {
            "nodes": selected_nodes,
            "mappers": {node.name: copy.deepcopy(node.drift_history[-1]) for node in selected_nodes}
        }
        self.concept_history.append(concept_snapshot)

    def local_drift(self):
        """Applies local drift to a single root node."""
        num_nodes_drifted = 1
        candidate_nodes = [v for v in self.vertices.values() if v.is_root()]

        selected_nodes = []

        for _ in range(num_nodes_drifted):
            drifted_node : Vertex = np.random.choice(candidate_nodes)
            selected_nodes.append(drifted_node)
            drifted_node.drift_history.append(copy.deepcopy(drifted_node.mapper)) 
            drifted_node.mapper.drift()

        if not hasattr(self, 'concept_history'):
            self.concept_history = []

        concept_snapshot = {
            "nodes": selected_nodes,
            "mappers": {node.name: copy.deepcopy(node.drift_history[-1]) for node in selected_nodes}
        }
        self.concept_history.append(concept_snapshot)

    def severe_drift(self) -> None:
        """Applies severe drift to the label node. Only works if the mapper supports severe_drift()."""
        label_node = self.vertices.get('y', None)

        if label_node is None:
            print("No label node 'y' found for severe drift.")
            return
        
        before_mapper = copy.deepcopy(label_node.mapper)

        label_node.drift_history.append(copy.deepcopy(before_mapper))

        if hasattr(label_node.mapper, 'severe_drift'):
            concept_snapshot = {
            "nodes": [label_node],
            "mappers": {node.name: copy.deepcopy(label_node.drift_history[-1]) for node in [label_node]}
        }
            self.concept_history.append(concept_snapshot)
            label_node.mapper.severe_drift()
            self.drifted_nodes = [label_node]
        else:
            print(f"Mapper {label_node.mapper} does not support severe drift.")

    def recurrent_drift(self) -> None:
        """Revert selected nodes to past concepts."""
        if not self.concept_history:
            print("No past concepts to return to.")
            return

        # Randomly select a past concept to return to
        concept = np.random.choice(self.concept_history)

        self.drifted_nodes = list(concept["nodes"])

        for node in self.drifted_nodes:
            target_past_mapper = copy.deepcopy(concept["mappers"][node.name])

            node.drift_history.append(copy.deepcopy(node.mapper))

            node.mapper = target_past_mapper


    def target_drift(self, incremental: bool = False) -> None:
        """Applies target drift to selected nodes. If incremental is True, applies incremental drift."""
        label_node = self.vertices.get('y', None)

        if label_node is None:
            print("No label node 'y' found for severe drift.")
            return
        
        before_mapper = copy.deepcopy(label_node.mapper)

        if not incremental:
            # abrupt drift
            label_node.mapper.drift()
        else:
            label_node.mapper.start_incremental_drift()

        # selected_nodes.append(drifted_node)
        self.drifted_nodes.append(label_node)
                
        label_node.drift_history.append(copy.deepcopy(before_mapper))

        if not hasattr(self, 'concept_history'):
            self.concept_history = []

        concept_snapshot = {
            "nodes": [label_node],
            "mappers": {node.name: copy.deepcopy(node.drift_history[-1]) for node in [label_node]}
        }
        self.concept_history.append(concept_snapshot)


    # Should finish this call
    # def counfounder_drift(self, incremental: bool = False) -> None:
    #     candidate_nodes = []

    #     target_node: Vertex = self.vertices.get('y', None)

    #     candidate_nodes = target_node.get_parents()

    #     # Candidate nodes should also be cause of other nodes.

    #     for node in candidate_nodes:
    #         if node.is_leaf():
    #             candidate_nodes.remove(node)
        
    #     # Confounder drift to only one node
    #     node_to_drift: Vertex = np.random.choice(candidate_nodes)

    #     # Get confounder drift idx
    #     confounder_idx = node_to_drift.children


    def simulate(self) -> None:
        """Computes values for all vertices in topological order."""
        for name in self.topological_sort():
            vertex = self.vertices[name]
            vertex.compute_value()

    

    def generate(self, dataset_size: int = 1000, intervention_prob: float = 0.05, drift_points: list = [], drift_sizes: list = [], drift_types: list = [], drift_types_time: list = [], missing_prob: float = 0.05) -> dict:
        """Generates samples from the causal graph with drift and interventions.
        Args:
            dataset_size (int): Number of samples to generate.
            intervention_prob (float): Probability of intervention on each sample.
            drift_points (list): List of points in time where drifts occur.
            drift_sizes (list): List of sizes of drifts corresponding to drift_points.
            drift_types (list): List of types of drifts ('real', 'virtual', etc.) corresponding to drift_points.
            drift_types_time (list): List of drift time types ('abrupt', 'gradual', etc.) corresponding to drift_points.
            missing_prob (float): Probability of missing values in each sample.
        Returns:
            dict: A dictionary with vertex names as keys and generated samples as values.
        """
        if not self.is_trained:
            self.train_graph()
        
        samples = {v.name: [] for v in self.vertices.values()}
        sorted_vertices = self.topological_sort()
        
        drift = 0
        drift_start = np.inf
        drift_end = -1

        for n in tqdm(range(dataset_size)):
            self.current_sample_index = n  # Track current sample index

            apply_intervention = np.random.rand() < intervention_prob
            intervened_nodes = set()

            apply_missing = np.random.rand() < missing_prob
            missing_nodes = set()
            
            if apply_intervention:
                num_intervened = np.random.randint(1, 4)  # 1 to 3 nodes
                intervened_nodes = set(np.random.choice(list(self.vertices.values()), size=num_intervened, replace=False))
                
            if apply_missing:
                num_missing = np.random.randint(1, 4)  # 1 to 3 nodes
                missing_nodes = set(np.random.choice(list(self.vertices.values()), size=num_missing, replace=False))
                
            
            if n in drift_points:
                drift_start = n
                drift_end = drift_points[drift] + drift_sizes[drift]
                drift_type_name = drift_types[drift]
                drift_time_type_name = drift_types_time[drift] if drift < len(drift_types_time) else 'abrupt'
                
                if drift_type_name == 'endogenous':
                    self.endogenous_drift(drift_time_type_name == 'incremental')
                    self._record_drift_event(n, 'endogenous', drift_time_type_name, self.drifted_nodes)
                elif drift_type_name == 'exogenous':
                    self.exogenous_drift()
                    self._record_drift_event(n, 'exogenous', drift_time_type_name, self.drifted_nodes)
                elif drift_type_name == 'recurrent':
                    self.recurrent_drift()
                    self._record_drift_event(n, 'recurrent', drift_time_type_name, self.drifted_nodes)
                elif drift_type_name == 'severe':
                    self.severe_drift()
                    self._record_drift_event(n, 'severe', drift_time_type_name, self.drifted_nodes)
                elif drift_type_name == 'target':
                    self.target_drift(drift_time_type_name == 'incremental')
                    self._record_drift_event(n, 'target', drift_time_type_name, self.drifted_nodes)

            for v in self.vertices.values():
                v.value = None

            for v_name in sorted_vertices:
                vtx : Vertex = self.vertices[v_name]
                
                if vtx in intervened_nodes and vtx.name != 'y':
                    self.assign_value(vtx, intervened_nodes)
                        
                else:
                    if not vtx.mapper.is_fitted():
                        vtx.mapper.fit(len(vtx.parents))

                    if vtx.is_root():
                        vtx.value = float(vtx.mapper.map(None))
                    else:
                        if n >= drift_start and n < drift_end and drift_types_time[drift] == 'gradual' and vtx in self.drifted_nodes:
                            if random.random() < 0.5:
                                vtx.compute_value_last_concept()
                            else:
                                vtx.compute_value()
                        elif n >= drift_start and n < drift_end and drift_types_time[drift] == 'incremental' and vtx in self.drifted_nodes:
                            vtx.mapper.partial_fit()
                            vtx.compute_value()
                        else:        
                            vtx.compute_value()

                samples[vtx.name].append(vtx.value)
                
            for v in missing_nodes:
                if v.name != 'y':
                    samples[v.name][n] = np.nan
            if n == drift_end:
                drift+=1
                drift_start = np.inf
                drift_end = -1

        for key in samples:
            samples[key] = np.array(samples[key])

        return samples
    
    def generate_sample(self, intervention_prob: float = 0) -> dict:
        """Generates a single sample from the causal graph with possible interventions.
        Args:
            intervention_prob (float): Probability of intervention on the sample.
        Returns:
            dict: A dictionary with vertex names as keys and generated sample values.
        """
        if not self.is_trained:
            self.train_graph()
        
        sample = {v.name: None for v in self.vertices.values()}
        sorted_vertices = self.topological_sort()
        
        apply_intervention = np.random.rand() < intervention_prob
        intervened_nodes = set()

        if apply_intervention:
            num_intervened = np.random.randint(1, 4)  # 1 to 3 nodes
            intervened_nodes = set(np.random.choice(list(self.vertices.values()), size=num_intervened, replace=False))
                
        for v in self.vertices.values():
            v.value = None

        for v_name in sorted_vertices:
            vtx : Vertex = self.vertices[v_name]
            
            if vtx in intervened_nodes and vtx.name != 'y':
                self.assign_value(vtx, intervened_nodes)
                    
            else:
                if not vtx.mapper.is_fitted():
                    vtx.mapper.fit(len(vtx.parents))

                if vtx.is_root():
                    vtx.value = float(vtx.mapper.map(None))
                else:
                    vtx.compute_value()

            sample[vtx.name] = vtx.value

        return sample
    
    def assign_value(self, node, intervened_nodes) -> None:
        """Ensure all parents are computed before assigning the value to the node."""
        if node.value is not None:
            return  # Already computed

        for parent in node.parents:
            if parent in intervened_nodes:
                self.assign_value(parent, intervened_nodes) 
            elif parent.value is None:
                if parent.is_root():
                    parent.value = float(parent.mapper.map(None))
                else:
                    self.assign_value(parent, intervened_nodes)

        if isinstance(node.mapper, AbstractCategoricalMapper):
            old_val = node.value
            while old_val == node.value:
                node.value = np.random.randint(0, 11) # 0 to 10 categories
        elif random.random() < 0.5:
            node.value = np.random.normal(0, 1)
        else:
            node.value = np.random.uniform(-5, 5)
    
    def visualize_graph(self, output: str = 'graph') -> None:
        """Visualizes the causal graph and saves it as an image."""
        G = nx.DiGraph()
        for v in self.vertices.values():
            for child in v.children:
                G.add_edge(v.name, child.name)
                
        node_colors = []
        for node in G.nodes():
            if self.vertices[node].name == 'y':
                node_colors.append("green")
            elif self.vertices[node].is_root():
                node_colors.append("red")            
            else:
                node_colors.append("lightblue")
                
        pos = nx.spring_layout(G) 
        # pos = nx.kamada_kawai_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_weight='bold')
        plt.savefig(f'{output}.png')
        plt.show()
        
    def save_graph(self, output: str = 'graph.pkl') -> None:
        """Save the CausalGraph object to a pickle file."""
        with open(output, 'wb') as f:                
            pickle.dump(self, f)
            
    def _record_drift_event(self, sample_index: int, drift_type: str, drift_time_type: str, drifted_nodes: list) -> None:
        """Record a drift event with detailed information about which nodes drifted and when.
        
        Args:
            sample_index (int): The sample index where the drift occurred.
            drift_type (str): Type of drift ('real', 'virtual', 'local', 'recurrent', 'severe').
            drift_time_type (str): Time type of drift ('abrupt', 'gradual', 'incremental').
            drifted_nodes (list): List of Vertex objects that drifted.
        """
        event = {
            "sample_index": sample_index,
            "drift_type": drift_type,
            "drift_time_type": drift_time_type,
            "drifted_nodes": [node.name for node in drifted_nodes],
            "node_details": []
        }
        
        # Collect details about each drifted node's mapper
        for node in drifted_nodes:
            node_detail = {
                "node_name": node.name,
                "mapper_type": str(type(node.mapper).__name__),
                "mapper_str": str(node.mapper),
            }
            
            # Try to capture mapper-specific details
            if hasattr(node.mapper, 'label_function'):
                node_detail["label_function"] = str(node.mapper.label_function)
            
            if isinstance(node.mapper, RandomMLPMapper):
                node_detail["mapper_class"] = "RandomMLPMapper"
                node_detail["hidden_dims"] = str(node.mapper.hidden_dims) if hasattr(node.mapper, 'hidden_dims') else "N/A"
                node_detail["activation"] = str(node.mapper.activation) if hasattr(node.mapper, 'activation') else "N/A"
            
            event["node_details"].append(node_detail)
        
        self.drift_events.append(event)
    
    def save_drift_events_to_json(self, output: str = 'drift_events.json') -> None:
        """Save all recorded drift events to a JSON file.
        
        Args:
            output (str): Path to output JSON file.
        """
        drift_data = {
            "total_drift_events": len(self.drift_events),
            "drift_events": self.drift_events
        }
        
        with open(output, 'w') as f:
            json.dump(drift_data, f, indent=4)
        
        print(f"Drift events saved to {output}")
    
    def get_drift_events_summary(self) -> dict:
        """Get a summary of all drift events.
        
        Returns:
            dict: Summary statistics about drift events.
        """
        summary = {
            "total_events": len(self.drift_events),
            "drift_types_count": {},
            "nodes_affected": {}
        }
        
        for event in self.drift_events:
            # Count drift types
            drift_type = event["drift_type"]
            summary["drift_types_count"][drift_type] = summary["drift_types_count"].get(drift_type, 0) + 1
            
            # Track which nodes were affected
            for node_name in event["drifted_nodes"]:
                summary["nodes_affected"][node_name] = summary["nodes_affected"].get(node_name, 0) + 1
        
        return summary

    def save_graph_to_json(self, output: str = 'graph.json', seed: int = None) -> None:
        """Saves the graph structure to a JSON file."""
        graph_data = {}

        if seed is not None:
            graph_data["seed"] = seed

        for node_name, node in self.vertices.items():
            parents = [parent.name for parent in node.parents]

            node_data = {
                'parents': parents,
                'mapper': str(node.mapper),
                'label_function': str(getattr(node.mapper, 'label_function', 'None'),),
            }
            graph_data[node_name] = node_data

        with open(output, 'w') as f:
            json.dump(graph_data, f, indent=4)
            
def load_graph(filepath: str = 'graph.pkl') -> CausalGraph:
    with open(filepath, 'rb') as f:
        return pickle.load(f)