import networkx as nx
import pandas as pd

def edges_to_operations_graph(edge_df: pd.DataFrame, nfeatures:int, nnodes:int) -> nx.DiGraph:
    original_graph = nx.from_pandas_edgelist(edge_df, 0, 1, 2, create_using=nx.DiGraph)
    for i in range(nfeatures):
        original_graph.add_edge(i,i)
    for i in range(nnodes - nfeatures, nnodes):
        original_graph.add_edge(i,i)

    H = nx.line_graph(original_graph, create_using=nx.DiGraph)
    H.add_nodes_from((node, original_graph.edges[node]) for node in H)
    
    for i in range(nfeatures):
        H.remove_edge((i,i), (i,i))
    for i in range(nnodes - nfeatures, nnodes):
        H.remove_edge((i,i), (i,i))

    return H

def operations_graph_to_edges(graph: nx.DiGraph) -> list[tuple]:
    pass