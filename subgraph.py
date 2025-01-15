import networkx as nx

class SubGraph:
    def __init__(self, graph: nx.Graph, info: int):
        self.graph = graph
        self.info = info