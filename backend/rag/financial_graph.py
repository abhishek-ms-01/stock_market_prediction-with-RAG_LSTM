import networkx as nx

class FinancialKnowledgeGraph:
    """
    IEEE Research Contribution: Knowledge Graph Integration for Financial Entity Context.
    Constructs entity-sector relationships using NetworkX.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()

    def _build_knowledge_graph(self):
        """Constructs knowledge graph nodes and directional edges."""
        edges = [
            # Ticker -> Subsidiary / Product -> Sector
            ("RELIANCE.NS", "Jio", "Subsidiary"),
            ("Jio", "5G Expansion", "Product"),
            ("5G Expansion", "Telecommunication", "Sector"),

            ("RELIANCE.NS", "Reliance Retail", "Subsidiary"),
            ("Reliance Retail", "E-Commerce", "Sector"),

            ("TCS.NS", "Tata Group", "Parent"),
            ("TCS.NS", "IT Services", "Sector"),

            ("INFY.NS", "Infosys Cobalt", "Product"),
            ("INFY.NS", "IT Services", "Sector"),

            ("HDFCBANK.NS", "HDFC Group", "Parent"),
            ("HDFCBANK.NS", "Banking & Finance", "Sector"),

            ("TATAMOTORS.NS", "Jaguar Land Rover", "Subsidiary"),
            ("TATAMOTORS.NS", "Automotive & EV", "Sector")
        ]

        for source, target, rel_type in edges:
            self.graph.add_edge(source, target, relation=rel_type)

    def get_connected_entities(self, entity: str) -> list:
        """Traverses the graph to retrieve related entity keywords."""
        if not entity or entity not in self.graph:
            return []
        
        # 1-hop and 2-hop graph neighbors
        neighbors = list(self.graph.successors(entity))
        for n in list(neighbors):
            neighbors.extend(list(self.graph.successors(n)))
        
        return list(set(neighbors))
