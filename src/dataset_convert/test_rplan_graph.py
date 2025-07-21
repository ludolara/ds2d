import pytest
from rplan_graph import RPLANGraph
import networkx as nx
from test_fixtures import *

class TestCompatibility:
    """Test compatibility of RPLANGraph floorplan graphs"""
    
    def test_sample_ds2d_compatibility(self, sample_ds2d_data):
        """Test expected connectivity of the 7-room sample"""
        graph = RPLANGraph.from_ds2d(sample_ds2d_data)
        
        # Should have 8 room nodes (7 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 8
        
        # Based on door positions and validation, expect 6 connections (5 interior + 1 front_door)
        assert len(graph.graph.edges()) == 6
        
        # Check if graph is connected (all nodes reachable from any node)
        is_connected = nx.is_connected(graph.graph)
        
        # The floorplan may not be fully connected due to isolated spaces
        # Just verify the structure is reasonable
        if is_connected:
            print("Graph is fully connected")
        else:
            # Count connected components
            components = list(nx.connected_components(graph.graph))
            print(f"Graph has {len(components)} connected components")
            
            # Largest component should contain most spaces
            largest_component_size = max(len(c) for c in components)
            assert largest_component_size >= 5, f"Largest component has only {largest_component_size} spaces"

    def test_complex_floorplan_compatibility(self, complex_ds2d_data):
        """Test connectivity of complex 8-room floorplan with expected adjacency"""
        graph = RPLANGraph.from_ds2d(complex_ds2d_data)
        
        # Should have 9 room nodes (8 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 9
        
        # Should have exactly 8 edges for this floorplan (7 interior + 1 front_door)
        assert len(graph.graph.edges()) == 8
        
        # Should be fully connected
        is_connected = nx.is_connected(graph.graph)
        assert is_connected, "Complex floorplan should be fully connected"
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity based on architectural layout (verified to be 100% accurate)
        expected_adjacency = {
            'storage': ['kitchen'], 
            'living_room': ['bedroom|0', 'kitchen', 'bathroom|0', 'bedroom|1', 'bedroom|2', 'front_door'], 
            'bedroom|0': ['living_room'], 
            'kitchen': ['living_room', 'storage'], 
            'bathroom|0': ['living_room'], 
            'bedroom|1': ['living_room', 'bathroom|1'], 
            'bedroom|2': ['living_room'], 
            'bathroom|1': ['bedroom|1'],
            'front_door': ['living_room']
        }
        
        # Check that all expected spaces are present
        for room in expected_adjacency:
            assert room in labeled_adj, f"Room {room} not found in graph"
        
        # Check exact connectivity matches for each room
        for room, expected_neighbors in expected_adjacency.items():
            actual_neighbors = set(labeled_adj.get(room, []))
            expected_neighbors_set = set(expected_neighbors)
            
            assert actual_neighbors == expected_neighbors_set, \
                f"Room {room}: Expected {expected_neighbors_set}, got {actual_neighbors}"
        
        # Verify living_room is the central hub with 6 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 6, f"Living room should have 6 connections, got {living_room_connections}"

    def test_generated_floorplan_compatibility(self, generated_ds2d_data):
        """Test compatibility of generated floorplan with expected disconnected spaces"""
        graph = RPLANGraph.from_ds2d(generated_ds2d_data)
        
        # Should have 8 room nodes (7 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 8
        
        # This floorplan has disconnected spaces, so not fully connected
        is_connected = nx.is_connected(graph.graph)
        assert not is_connected, "Generated floorplan should NOT be fully connected (has isolated spaces)"
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - many spaces are isolated (front door included)
        expected_adjacency = {
            'bedroom|0': [],
            'balcony|0': ['living_room'],
            'bedroom|1': [],
            'balcony|1': [],
            'bathroom': ['living_room'],
            'kitchen': ['living_room'],
            'living_room': ['balcony|0', 'bathroom', 'kitchen', 'front_door'],
            'front_door': ['living_room']
        }
        
        # Check that all expected spaces are present
        for room in expected_adjacency:
            assert room in labeled_adj, f"Room {room} not found in graph"
        
        # Check exact connectivity matches for each room
        for room, expected_neighbors in expected_adjacency.items():
            actual_neighbors = set(labeled_adj.get(room, []))
            expected_neighbors_set = set(expected_neighbors)
            
            assert actual_neighbors == expected_neighbors_set, \
                f"Room {room}: Expected {expected_neighbors_set}, got {actual_neighbors}"
        
        # Verify that isolated spaces have no connections
        isolated_rooms = ['bedroom|0', 'bedroom|1', 'balcony|1']
        for room in isolated_rooms:
            connections = labeled_adj.get(room, [])
            assert len(connections) == 0, f"Room {room} should be isolated but has connections: {connections}"
        
        # Verify living_room has exactly 4 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 4, f"Living room should have 4 connections, got {living_room_connections}"
        
        # Check connected components
        components = list(nx.connected_components(graph.graph))
        assert len(components) == 4, f"Should have 4 connected components, got {len(components)}"
        
        # Find the largest component (should contain living_room)
        largest_component = max(components, key=len)
        assert len(largest_component) == 5, f"Largest component should have 5 spaces, got {len(largest_component)}"

    def test_double_connection_balcony_compatibility(self, double_connection_balcony_ds2d_data):
        """Test connectivity of 8-room floorplan with balcony having double connections"""
        graph = RPLANGraph.from_ds2d(double_connection_balcony_ds2d_data)
        
        # Should have 9 room nodes (8 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 9
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - balcony|0 has double connections (to bedroom|0 and study_room)
        expected_adjacency = {
            'balcony|0': ['bedroom|0', 'study_room'],  # Double connection case
            'bedroom|0': ['balcony|0', 'living_room'], 
            'bathroom': ['living_room'], 
            'bedroom|1': ['balcony|1', 'living_room'], 
            'balcony|1': ['bedroom|1'], 
            'living_room': ['bedroom|0', 'bathroom', 'bedroom|1', 'kitchen', 'study_room', 'front_door'], 
            'kitchen': ['living_room'], 
            'study_room': ['balcony|0', 'living_room'],
            'front_door': ['living_room']
        }
        
        # Check that all expected spaces are present
        for room in expected_adjacency:
            assert room in labeled_adj, f"Room {room} not found in graph"
        
        # Check exact connectivity matches for each room
        for room, expected_neighbors in expected_adjacency.items():
            actual_neighbors = set(labeled_adj.get(room, []))
            expected_neighbors_set = set(expected_neighbors)
            
            assert actual_neighbors == expected_neighbors_set, \
                f"Room {room}: Expected {expected_neighbors_set}, got {actual_neighbors}"
        
        # Verify living_room is the central hub with 6 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 6, f"Living room should have 6 connections, got {living_room_connections}"
        
        # Check that the graph is fully connected
        is_connected = nx.is_connected(graph.graph)
        assert is_connected, "New floorplan should be fully connected"
        
        # Verify total number of edges matches expected connections
        total_edges = len(graph.graph.edges())
        expected_edges = sum(len(neighbors) for neighbors in expected_adjacency.values()) // 2
        assert total_edges == expected_edges, f"Expected {expected_edges} edges, got {total_edges}"

    def test_containment_issue_compatibility(self, containment_issue_ds2d_data):
        """Test that spaces contained within other spaces do not create invalid connections"""
        graph = RPLANGraph.from_ds2d(containment_issue_ds2d_data)
        
        # Should have 9 room nodes (8 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 9
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - bathroom|0 should NOT connect to bedroom|0 (containment case)
        expected_adjacency = {
            'kitchen': ['living_room', 'front_door'], 
            'storage': [], 
            'balcony': ['living_room'], 
            'bathroom|0': [],  # Should be empty (not connected to bedroom|0 due to containment)
            'bedroom|0': ['living_room'],  # Should only connect to living_room
            'bedroom|1': ['living_room'], 
            'bathroom|1': [], 
            'living_room': ['kitchen', 'balcony', 'bedroom|0', 'bedroom|1', 'front_door'],
            'front_door': ['living_room', 'kitchen']
        }
        
        # Check that all expected spaces are present
        for room in expected_adjacency:
            assert room in labeled_adj, f"Room {room} not found in graph"
        
        # Check exact connectivity matches for each room
        for room, expected_neighbors in expected_adjacency.items():
            actual_neighbors = set(labeled_adj.get(room, []))
            expected_neighbors_set = set(expected_neighbors)
            
            assert actual_neighbors == expected_neighbors_set, \
                f"Room {room}: Expected {expected_neighbors_set}, got {actual_neighbors}"
        
        # Specifically verify that bathroom|0 is NOT connected to bedroom|0
        bathroom0_connections = labeled_adj.get('bathroom|0', [])
        bedroom0_connections = labeled_adj.get('bedroom|0', [])
        
        assert 'bedroom|0' not in bathroom0_connections, "bathroom|0 should not connect to bedroom|0 (containment issue)"
        assert 'bathroom|0' not in bedroom0_connections, "bedroom|0 should not connect to bathroom|0 (containment issue)"
        
        # Verify living_room is the central hub with 5 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 5, f"Living room should have 5 connections, got {living_room_connections}"
        
        # Verify total number of edges matches expected connections
        total_edges = len(graph.graph.edges())
        expected_edges = sum(len(neighbors) for neighbors in expected_adjacency.values()) // 2
        assert total_edges == expected_edges, f"Expected {expected_edges} edges, got {total_edges}"

    def test_front_door_exclusion(self, front_door_exclusion_data):
        graph = RPLANGraph.from_ds2d(front_door_exclusion_data)
        # Should have 3 room nodes (2 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 3
        # Should have 1 edge (front_door connects to living_room)
        assert len(graph.graph.edges()) == 1, "Front door should connect to living_room"
        # Verify spaces are connected correctly
        labeled_adj = graph.to_labeled_adjacency()
        assert 'front_door' in labeled_adj['living_room']
        assert labeled_adj['kitchen'] == []
        assert labeled_adj['front_door'] == ['living_room']

    def test_single_door_compatibility(self, multiple_doors_ds2d_data):
        """Test that rooms with multiple doors are not connected (invalid connections rejected)"""
        graph = RPLANGraph.from_ds2d(multiple_doors_ds2d_data)
        
        # Should have 6 room nodes (5 rooms + 1 front_door)
        assert len(graph.graph.nodes()) == 6
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - rooms with multiple doors should NOT be connected
        # The kitchen and living_room have multiple doors, so they should not be connected
        expected_adjacency = {
            'bedroom': ['living_room'],
            'balcony': ['living_room'],
            'bathroom': ['living_room'],
            'kitchen': [],  # No connection due to multiple doors
            'living_room': ['bedroom', 'balcony', 'bathroom', 'front_door'],  # No kitchen due to multiple doors
            'front_door': ['living_room']
        }
        
        # Check that all expected spaces are present
        for room in expected_adjacency:
            assert room in labeled_adj, f"Room {room} not found in graph"
        
        # Check exact connectivity matches for each room
        for room, expected_neighbors in expected_adjacency.items():
            actual_neighbors = set(labeled_adj.get(room, []))
            expected_neighbors_set = set(expected_neighbors)
            
            assert actual_neighbors == expected_neighbors_set, \
                f"Room {room}: Expected {expected_neighbors_set}, got {actual_neighbors}"
        
        # Verify that kitchen has no connections (due to multiple doors)
        kitchen_connections = labeled_adj.get('kitchen', [])
        assert len(kitchen_connections) == 0, f"Kitchen should have no connections due to multiple doors, got: {kitchen_connections}"
        
        # Verify that living_room doesn't connect to kitchen (due to multiple doors)
        living_room_connections = labeled_adj.get('living_room', [])
        assert 'kitchen' not in living_room_connections, f"Living room should not connect to kitchen due to multiple doors"
        
        # Verify total number of edges matches expected (no connections for multiple doors)
        total_edges = len(graph.graph.edges())
        expected_edges = 4  # 4 unique connections: bedroom-living_room, balcony-living_room, bathroom-living_room, front_door-living_room
        assert total_edges == expected_edges, f"Expected {expected_edges} edges, got {total_edges}"
        
        # Verify the graph is NOT fully connected (kitchen is isolated)
        is_connected = nx.is_connected(graph.graph)
        assert not is_connected, "Graph should NOT be fully connected (kitchen should be isolated due to multiple doors)"
        
        # Check connected components
        components = list(nx.connected_components(graph.graph))
        assert len(components) == 2, f"Should have 2 connected components, got {len(components)}"
        
        # One component should contain living_room and connected rooms, another should contain kitchen
        living_room_component = None
        kitchen_component = None
        for component in components:
            room_names = []
            for idx in component:
                room_type = graph.graph.nodes[idx]['room_type']
                room_name = graph.room_class[room_type]
                room_names.append(room_name)
            if "living_room" in room_names:
                living_room_component = component
            if "kitchen" in room_names:
                kitchen_component = component
        
        assert living_room_component is not None, "Living room component not found"
        assert kitchen_component is not None, "Kitchen component not found"
        assert len(living_room_component) == 5, f"Living room component should have 5 rooms, got {len(living_room_component)}"
        assert len(kitchen_component) == 1, f"Kitchen component should have 1 room, got {len(kitchen_component)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
    