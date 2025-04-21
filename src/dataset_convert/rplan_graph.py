from dataclasses import dataclass, field
import networkx as nx
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Set
from collections import Counter, defaultdict

@dataclass
class RPLANGraph:
    """
    Represent a floorplan as a graph, excluding doors from nodes and edges.

    Attributes:
        floorplan: Original floorplan dict with "room_type" and "ed_rm".
        door_idxs: Set of indices corresponding to doors.
        graph: NetworkX graph of rooms (doors excluded).
        adj: Adjacency list mapping each room node to its neighbors.
    """
    floorplan: Dict[str, Any]
    room_class = {
        1:  "living_room",
        2:  "kitchen",
        3:  "bedroom",
        4:  "bathroom",
        5:  "balcony",
        6:  "entrance",
        7:  "dining_room",
        8:  "study_room",
        10: "storage",
        15: "front_door",
        16: "unknown",
        17: "interior_door",
    }
    room_door_types: Set[int] = field(default_factory=lambda: {15, 17})
    cmap: Dict[int, str] = field(default_factory=lambda: {
        1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171',
        12: '#D3A2C7', 13: '#785A67'
    })
    door_idxs: Set[int] = field(init=False)
    graph: nx.Graph = field(init=False)
    adj: Dict[int, List[int]] = field(init=False)

    def __post_init__(self):
        # Identify door indices
        self.door_idxs = {
            i for i, c in enumerate(self.floorplan.get("room_type", []))
            if c in self.room_door_types
        }
        # Build graph excluding doors
        self.graph = nx.Graph()
        for i, c in enumerate(self.floorplan.get("room_type", [])):
            if i not in self.door_idxs:
                self.graph.add_node(i, room_type=c)

        for edge in self.floorplan.get("ed_rm", []):
            if len(edge) == 2:
                a, b = edge
                if a not in self.door_idxs and b not in self.door_idxs:
                    self.graph.add_edge(a, b)

        self.adj = {n: list(self.graph.neighbors(n)) for n in self.graph.nodes()}
        self.bubble_diagram = self._compute_bubble_diagram()

    def _compute_bubble_diagram(self) -> Dict[str, List[str]]:
        """
        Generate a labeled adjacency dict (bubble diagram) where each room node is labeled by
        its type and a sequential index if duplicates exist, mapping to similarly labeled neighbors.
        """
        raw_adj = {n: list(self.graph.neighbors(n)) for n in self.graph.nodes()}
        counts = Counter(self.graph.nodes[n]['room_type'] for n in self.graph.nodes())
        type_seq: Dict[int, int] = defaultdict(int)
        type_idx: Dict[int, Any] = {}

        for idx in sorted(self.graph.nodes):
            rt = self.graph.nodes[idx]['room_type']
            if counts[rt] > 1:
                type_idx[idx] = type_seq[rt]
                type_seq[rt] += 1
            else:
                type_idx[idx] = None

        labeled: Dict[str, List[str]] = {}
        for idx, nbrs in raw_adj.items():
            rt = self.graph.nodes[idx]['room_type']
            base = self.room_class[rt]
            src_label = f"{base}_{type_idx[idx]}" if type_idx[idx] is not None else base
            neighbor_labels: List[str] = []
            for n in nbrs:
                rt_n = self.graph.nodes[n]['room_type']
                nb_base = self.room_class[rt_n]
                nb_label = f"{nb_base}_{type_idx[n]}" if type_idx[n] is not None else nb_base
                neighbor_labels.append(nb_label)
            labeled[src_label] = neighbor_labels
        return labeled

    def draw(self, title: str = "Floorplan Graph") -> None:
        """
        Draw the floorplan graph using matplotlib.
        Nodes colored by room_type per cmap.
        """
        pos = nx.spring_layout(self.graph, seed=84)
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=[self.cmap[self.graph.nodes[n]["room_type"]] for n in self.graph.nodes()],
            node_size=300,
        )
        nx.draw_networkx_edges(self.graph, pos, width=2)
        plt.title(title)
        plt.axis("off")

# if __name__ == "__main__":
#     fg = FloorplanGraph(floorplan)
#     print("Adjacency list:", fg.adj)
#     fg.draw("Input Graph")
#     plt.show()
    