from SCM.Mappers import *
import SCM.CausalGraph as cg
import random

import numpy as np
from collections import deque

categorical_mappers = [PrototypeCategoricalMapper]
reg_mappers = [RandomMLPMapper, FunctionMapper]

def get_mapper_lists():    
    root_mappers = [NormalMapper, UniformMapper]
    edge_mappers = [
        PrototypeCategoricalMapper, RandomMLPMapper, FunctionMapper
    ]
    return root_mappers, edge_mappers

def build_er_dag(n_features, p_edge=0.3, task='classification', ensure_label_parents=True, n_confounders=0, create_confounder=False, seed=42):
    """
    Builds an Erdős–Rényi DAG.
    
    Parameters
    ----------
    n_features : int
        Number of X features (label will be added automatically).
    p_edge : float
        Probability of edge between any ordered pair.
    task : str
        'classification' or 'regression'
    ensure_label_parents : bool
        If True, guarantees label has at least one parent.
    n_confounders : int
        Number of confounder nodes to add (only if create_confounder=True).
    create_confounder : bool
        If True, creates confounder nodes that are parents of the label and some other nodes
    """

    root_mappers, edge_mappers = get_mapper_lists()
    n_features += 1  # +1 for label
    random.seed(seed)

    nodes = [cg.Vertex(f"x{i}") for i in range(n_features)]

    # Randomly choose label
    label_index = random.randint(0, n_features - 1)
    label_node = nodes[label_index]
    label_node.name = "y"

    # Random topological order
    node_order = list(range(n_features))
    random.shuffle(node_order)

    graph = cg.CausalGraph()
    for node in nodes:
        graph.add_vertex(node)

    # Add ER edges (only forward in order to ensure DAG)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if random.random() < p_edge:
                parent = nodes[node_order[i]]
                child = nodes[node_order[j]]
                graph.add_edge(parent, child)

    if ensure_label_parents and label_node.is_root():
        possible_parents = [nodes[i] for i in node_order if nodes[i] != label_node]
        parent = random.choice(possible_parents)
        graph.add_edge(parent, label_node)

    if create_confounder:
        # Create n confounders that are parents of both y and a random subset of other nodes
        for i in range(n_confounders):
            confounder = cg.Vertex(f"c{i}")
            graph.add_vertex(confounder)
            graph.add_edge(confounder, label_node)

            # Connect confounder to a random subset of other nodes (excluding label)
            other_nodes = [n for n in nodes if n != label_node]
            num_edges = random.randint(1, max(1, len(other_nodes) // 2))
            for target in random.sample(other_nodes, num_edges):
                graph.add_edge(confounder, target)
            nodes.append(confounder)

    # Assign mappers
    for node in nodes:
        if node.is_root():
            node.mapper = random.choice(root_mappers)()
        elif node is label_node:
            if task == 'regression':
                node.mapper = random.choice(reg_mappers)()
            else:
                node.mapper = random.choice(categorical_mappers)()
        else:
            node.mapper = random.choice(edge_mappers)()

    return graph, label_node

def build_connected_dag(n_features, n_roots, max_parents, task='classification'):
    root_mappers, edge_mappers = get_mapper_lists()
    n_features+=1
    
    nodes = [cg.Vertex(f"x{i}") for i in range(n_features)]
    
    root_indices = random.sample(range(n_features), k=n_roots)
    roots = [nodes[i] for i in root_indices]
    
    label_candidates = [i for i in range(n_features) if i not in root_indices]
    label_index = random.choice(label_candidates)
    label_node = nodes[label_index]
    label_node.name = 'y'
    
    node_order = list(range(n_features))
    random.shuffle(node_order)
    
    if len(root_indices) >= 2:
        roots_to_place = random.sample(root_indices, 2)
        
        for i in range(2):
            root = roots_to_place[i]
            idx_to_swap = node_order.index(root)
            node_order[i], node_order[idx_to_swap] = node_order[idx_to_swap], node_order[i]

    elif len(root_indices) == 1:
        root = root_indices[0]
        idx_to_swap = node_order.index(root)
        node_order[0], node_order[idx_to_swap] = node_order[idx_to_swap], node_order[0]
    
    graph = cg.CausalGraph()
    for node in nodes:
        graph.add_vertex(node)
    
    for i, idx in enumerate(node_order):
        node = nodes[idx]
        if node in roots:
            continue

        candidate_parents = [nodes[j] for j in node_order[:i]]
        
        max_parents_this_node = min(max_parents, len(candidate_parents))
        if (max_parents_this_node < 1):
            raise ValueError("Max parents can't be less than 1")
        if node is label_node:
            n_parents = max(2, random.randint(1, max_parents_this_node))
        else:
            n_parents = random.randint(1, max_parents_this_node)
            n_parents = max(1, n_parents)
        parents = random.sample(candidate_parents, n_parents)
        for p in parents:
            graph.add_edge(p, node)
            
    for node in nodes:
        if node.is_root():
            node.mapper = random.choice(root_mappers)()
        elif node is label_node:
            if task == 'regression':
                node.mapper = np.random.choice(reg_mappers)()
            else:
                node.mapper = np.random.choice(categorical_mappers)()
        else:
            node.mapper = random.choice(edge_mappers)()
    
    return graph, label_node

def extract_graph_statistics(graph):
    """
    Extract structural and causal statistics from your CausalGraph class.
    
    Assumes label node is named 'y'.
    
    Returns:
        dict of statistics
    """

    stats = {}

    vertices = list(graph.vertices.values())
    n = len(vertices)

    stats["num_nodes"] = n

    # -----------------------
    # Basic structural stats
    # -----------------------
    num_edges = 0
    in_degrees = []
    out_degrees = []
    roots = []

    for v in vertices:
        in_deg = len(v.parents)
        out_deg = len(v.children)

        in_degrees.append(in_deg)
        out_degrees.append(out_deg)
        num_edges += out_deg

        if v.is_root():
            roots.append(v)

    stats["num_edges"] = num_edges
    stats["density"] = num_edges / (n * (n - 1) / 2) if n > 1 else 0

    stats["avg_in_degree"] = float(np.mean(in_degrees))
    stats["avg_out_degree"] = float(np.mean(out_degrees))
    stats["num_roots"] = len(roots)

    # -----------------------
    # Label-specific stats
    # -----------------------
    label_node = graph.vertices.get("y", None)

    if label_node is None:
        raise ValueError("No label node named 'y' found.")

    parents_y = label_node.parents
    children_y = label_node.children

    stats["parents_y"] = len(parents_y)
    stats["children_y"] = len(children_y)

    # -----------------------
    # Ancestors of y
    # -----------------------
    def get_ancestors(node):
        visited = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for p in current.parents:
                if p not in visited:
                    visited.add(p)
                    stack.append(p)
        return visited

    ancestors_y = get_ancestors(label_node)
    stats["ancestors_y"] = len(ancestors_y)

    # -----------------------
    # Descendants of y
    # -----------------------
    def get_descendants(node):
        visited = set()
        stack = [node]
        while stack:
            current = stack.pop()
            for c in current.children:
                if c not in visited:
                    visited.add(c)
                    stack.append(c)
        return visited

    descendants_y = get_descendants(label_node)
    stats["descendants_y"] = len(descendants_y)

    # -----------------------
    # Markov blanket of y
    # -----------------------
    spouses = set()
    for child in children_y:
        for p in child.parents:
            if p != label_node:
                spouses.add(p)

    markov_blanket_y = set(parents_y) | set(children_y) | spouses
    stats["markov_blanket_y"] = len(markov_blanket_y)

    # -----------------------
    # Non-ancestors of y
    # -----------------------
    non_ancestors = set(vertices) - ancestors_y - {label_node}
    stats["non_ancestors_y"] = len(non_ancestors)

    # -----------------------
    # Longest path (DAG depth)
    # -----------------------
    topo_order_names = graph.topological_sort()
    topo_vertices = [graph.vertices[name] for name in topo_order_names]

    # Dynamic programming for longest path
    dist = {v: 0 for v in topo_vertices}

    for v in topo_vertices:
        for child in v.children:
            dist[child] = max(dist[child], dist[v] + 1)

    stats["global_longest_path"] = max(dist.values())
    stats["longest_path_to_y"] = dist[label_node]

    # -----------------------
    # Difficulty proxies
    # -----------------------
    stats["ancestor_ratio_y"] = stats["ancestors_y"] / n
    stats["mb_ratio_y"] = stats["markov_blanket_y"] / n

    # -----------------------
    # Drift-related structure
    # -----------------------
    if hasattr(graph, "drift_events") and len(graph.drift_events) > 0:

        total_drifted_nodes = set()
        drift_on_ancestors = 0
        drift_on_non_ancestors = 0

        for event in graph.drift_events:
            for node_name in event["drifted_nodes"]:
                total_drifted_nodes.add(node_name)

                node_obj = graph.vertices[node_name]

                if node_obj in ancestors_y:
                    drift_on_ancestors += 1
                elif node_obj != label_node:
                    drift_on_non_ancestors += 1

        stats["total_unique_drifted_nodes"] = len(total_drifted_nodes)
        stats["drift_on_ancestors_y"] = drift_on_ancestors
        stats["drift_on_non_ancestors_y"] = drift_on_non_ancestors
    else:
        stats["total_unique_drifted_nodes"] = 0
        stats["drift_on_ancestors_y"] = 0
        stats["drift_on_non_ancestors_y"] = 0

    return stats
