from dataclasses import dataclass, field
import networkx as nx
# import matplotlib.pyplot as plt
from typing import Any, Dict, List, Set
from collections import Counter, defaultdict
from shapely.geometry import Polygon
from itertools import combinations

ROOM_CLASS = {
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

CMAP = {
  1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8', 6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 10: '#1F849B', 11: '#727171', 12: '#D3A2C7', 13: '#785A67'
}

@dataclass
class RPLANGraph:
    """
    Represent a floorplan as a graph.
    """
    floorplan: Dict[str, Any]
    room_class: Dict[int,str] = field(default_factory=lambda: ROOM_CLASS.copy())
    cmap: Dict[int,str] = field(default_factory=lambda: CMAP.copy())
    room_door_types: Set[int] = frozenset({15,17})
    door_idxs: Set[int] = field(init=False)
    graph: nx.Graph = field(init=False)
    # adj: Dict[int, List[int]] = field(init=False)

    @classmethod
    def from_housegan(cls, fp: Dict[str,Any]) -> "RPLANGraph":
        inst = cls(fp)

        types = fp.get("room_type", [])
        inst.door_idxs = {
            idx for idx, t in enumerate(types)
            if t in inst.room_door_types
        }

        G = nx.Graph()
        for idx, t in enumerate(types):
            if idx not in inst.door_idxs:
                G.add_node(idx, room_type=t)

        for edge in fp.get("ed_rm", []):
            if len(edge) == 2:
                a, b = edge
                if a not in inst.door_idxs and b not in inst.door_idxs:
                    G.add_edge(a, b)

        inst.graph = G
        # inst.adj = {n: list(G.neighbors(n)) for n in G.nodes()}
        return inst

    @classmethod
    def from_ds2d(cls, data: Dict[str,Any], gap_threshold: float = 0.2, boundary_tolerance: float = 0.3, overlap_threshold: float = 0.3) -> "RPLANGraph":
        inst = cls({})
        room_polys = {}
        door_polys = {}
        name_to_int = {v:k for k,v in ROOM_CLASS.items()}
        
        # Validate data structure
        if not isinstance(data, dict) or "spaces" not in data:
            print(f"ERROR: from_ds2d data is malformed: {data}")
            inst.door_idxs = set()
            inst.graph = nx.Graph()
            return inst
            
        spaces = data["spaces"]
        if not isinstance(spaces, list):
            print(f"ERROR: from_ds2d spaces is not a list: {spaces}")
            inst.door_idxs = set()
            inst.graph = nx.Graph()
            return inst
        
        for idx, item in enumerate(spaces):
            # Validate item is a dictionary
            if not isinstance(item, dict):
                print(f"ERROR: from_ds2d room {idx} is not a dict: {item}")
                continue
                
            # Validate required keys exist
            if "floor_polygon" not in item or "room_type" not in item:
                print(f"ERROR: from_ds2d room {idx} missing required keys: {item}")
                continue
                
            floor_polygon = item["floor_polygon"]
            if not isinstance(floor_polygon, list):
                print(f"ERROR: from_ds2d room {idx} floor_polygon is not a list: {floor_polygon}")
                continue
            
            try:
                coords = []
                for pt_idx, pt in enumerate(floor_polygon):
                    if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                        print(f"ERROR: from_ds2d room {idx} point {pt_idx} is malformed: {pt}")
                        break
                    coords.append((pt["x"], pt["y"]))
                
                if len(coords) != len(floor_polygon):
                    continue  # Skip this room if any points were invalid
                    
                poly = Polygon(coords)
                if item["room_type"] in ("interior_door", "front_door"):
                    door_polys[idx] = poly
                else:
                    room_polys[idx] = poly
            except Exception as e:
                print(f"ERROR: from_ds2d processing room {idx}: {e}")
                continue
        inst.door_idxs = set(door_polys.keys())
        G = nx.Graph()
        for idx in room_polys:
            rt_int = name_to_int.get(data["spaces"][idx]["room_type"], name_to_int["unknown"])
            G.add_node(idx, room_type=rt_int)
        
        for door_idx, door_poly in door_polys.items():
            door_room_type = data["spaces"][door_idx]["room_type"]
            if door_room_type == "front_door":
                continue
                
            buf = door_poly.buffer(gap_threshold)
            touching = [i for i,p in room_polys.items() if buf.intersects(p)]
            
            for a,b in combinations(touching, 2):
                # Validation 1: Check if the door is close to the gap between the two spaces
                room_a_poly = room_polys[a]
                room_b_poly = room_polys[b]
                
                # Check if door is close to both room boundaries (more tolerant approach)
                door_centroid = door_poly.centroid
                dist_to_a = room_a_poly.boundary.distance(door_centroid)
                dist_to_b = room_b_poly.boundary.distance(door_centroid)
                
                # Door should be close to both room boundaries
                if dist_to_a > boundary_tolerance or dist_to_b > boundary_tolerance:
                    continue  # Skip if door is too far from either room boundary
                
                # Validation 2: Check if door significantly overlaps with room interiors
                door_area = door_poly.area
                
                valid_connection = True
                for room_poly in room_polys.values():
                    if door_poly.overlaps(room_poly):
                        overlap_area = door_poly.intersection(room_poly).area
                        overlap_ratio = overlap_area / door_area if door_area > 0 else 0
                        if overlap_ratio > overlap_threshold:
                            valid_connection = False
                            break
                
                if not valid_connection:
                    continue  # Skip this connection if door overlaps significantly with any room
                
                # Validation 3: Check if one room is contained within the other (invalid connection)
                if room_a_poly.contains(room_b_poly) or room_b_poly.contains(room_a_poly):
                    continue  # Skip if one room is completely inside the other
                
                G.add_edge(a, b)
        
        inst.graph = G
        return inst

    @classmethod
    def from_labeled_adjacency(cls, labeled: Dict[str, List[str]]) -> "RPLANGraph":
        inst = cls({})
        G = nx.Graph()
        name_to_int = {v: k for k, v in ROOM_CLASS.items()}
        label_to_idx: Dict[str, int] = {}
        for idx, label in enumerate(labeled):
            label_to_idx[label] = idx
            base = label.split("|", 1)[0]
            rt_int = name_to_int.get(base, name_to_int["unknown"])
            G.add_node(idx, room_type=rt_int)
        for src_label, nbr_labels in labeled.items():
            a = label_to_idx[src_label]
            for nb_label in nbr_labels:
                b = label_to_idx[nb_label]
                G.add_edge(a, b)
        inst.graph = G
        inst.door_idxs = set()
        return inst

    def to_labeled_adjacency(self) -> Dict[str, List[str]]:
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
            src_label = f"{base}|{type_idx[idx]}" if type_idx[idx] is not None else base
            neighbor_labels: List[str] = []
            for n in nbrs:
                rt_n = self.graph.nodes[n]['room_type']
                nb_base = self.room_class[rt_n]
                nb_label = f"{nb_base}|{type_idx[n]}" if type_idx[n] is not None else nb_base
                neighbor_labels.append(nb_label)
            labeled[src_label] = neighbor_labels
        return labeled

    
    def compatibility_score(self, other: "RPLANGraph") -> int:
        def multiset_edges(graph: nx.Graph) -> Counter:
            cnt = Counter()
            for u, v in graph.edges():
                rt_u = graph.nodes[u]['room_type']
                rt_v = graph.nodes[v]['room_type']
                base_u = self.room_class.get(rt_u, 'unknown')
                base_v = self.room_class.get(rt_v, 'unknown')
                edge = tuple(sorted((base_u, base_v)))
                cnt[edge] += 1
            return cnt

        c1 = multiset_edges(self.graph)
        c2 = multiset_edges(other.graph)
        all_edges = set(c1.keys()) | set(c2.keys())
        mistakes = sum(abs(c1[e] - c2[e]) for e in all_edges)
        return mistakes
    
    def compatibility_score_scaled(self, other: "RPLANGraph") -> float:
        def multiset_edges(graph: nx.Graph) -> Counter:
            cnt = Counter()
            for u, v in graph.edges():
                rt_u = graph.nodes[u]['room_type']
                rt_v = graph.nodes[v]['room_type']
                base_u = self.room_class.get(rt_u, 'unknown')
                base_v = self.room_class.get(rt_v, 'unknown')
                edge = tuple(sorted((base_u, base_v)))
                cnt[edge] += 1
            return cnt

        c1 = multiset_edges(self.graph)
        c2 = multiset_edges(other.graph)
        all_edges = set(c1.keys()) | set(c2.keys())

        mismatches = sum(abs(c1[e] - c2[e]) for e in all_edges)
        total = sum(max(c1[e], c2[e]) for e in all_edges)
        if total == 0:
            return 1.0
        return 1.0 - mismatches / total
        
    # def draw(self, title: str = "Input Graph", seed: int = 42) -> None:
    #     pos = nx.spring_layout(self.graph, seed=seed)
    #     nx.draw_networkx_nodes(
    #         self.graph,
    #         pos,
    #         node_color=[self.cmap[self.graph.nodes[n]["room_type"]] for n in self.graph.nodes()],
    #         node_size=300,
    #     )
    #     nx.draw_networkx_edges(self.graph, pos, width=2)
    #     plt.title(title)
    #     plt.axis("off")
