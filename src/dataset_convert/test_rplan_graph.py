import pytest
from rplan_graph import RPLANGraph
import networkx as nx

@pytest.fixture
def sample_ds2d_data():
    """Fixture providing sample DS2D data for testing"""
    return {
        "room_count": 7,
        "spaces": [
            {
                "id": "bedroom|0",
                "room_type": "bedroom",
                "area": 13.2,
                "floor_polygon": [
                    {"x": 5.9, "y": 7.3},
                    {"x": 8.7, "y": 7.3},
                    {"x": 8.7, "y": 2.6},
                    {"x": 5.9, "y": 2.6}
                ]
            },
            {
                "id": "storage",
                "room_type": "storage",
                "area": 1.3,
                "floor_polygon": [
                    {"x": 7.8, "y": 8.3},
                    {"x": 7.8, "y": 7.6},
                    {"x": 5.9, "y": 7.6},
                    {"x": 5.9, "y": 8.3}
                ]
            },
            {
                "id": "bedroom|1",
                "room_type": "bedroom",
                "area": 14.8,
                "floor_polygon": [
                    {"x": 5.9, "y": 8.6},
                    {"x": 5.9, "y": 13.9},
                    {"x": 8.7, "y": 13.9},
                    {"x": 8.7, "y": 8.6}
                ]
            },
            {
                "id": "kitchen",
                "room_type": "kitchen",
                "area": 5.4,
                "floor_polygon": [
                    {"x": 10.8, "y": 5.7},
                    {"x": 10.8, "y": 2.6},
                    {"x": 9.0, "y": 2.6},
                    {"x": 9.0, "y": 5.7}
                ]
            },
            {
                "id": "bathroom",
                "room_type": "bathroom",
                "area": 1.7,
                "floor_polygon": [
                    {"x": 11.1, "y": 4.5},
                    {"x": 12.0, "y": 4.5},
                    {"x": 12.0, "y": 2.6},
                    {"x": 11.1, "y": 2.6}
                ]
            },
            {
                "id": "living_room",
                "room_type": "living_room",
                "area": 25.4,
                "floor_polygon": [
                    {"x": 9.0, "y": 7.6},
                    {"x": 8.1, "y": 7.6},
                    {"x": 8.1, "y": 8.3},
                    {"x": 9.0, "y": 8.3},
                    {"x": 9.0, "y": 13.9},
                    {"x": 12.0, "y": 13.9},
                    {"x": 12.0, "y": 4.8},
                    {"x": 11.1, "y": 4.8},
                    {"x": 11.1, "y": 6.0},
                    {"x": 9.0, "y": 6.0}
                ]
            },
            {
                "id": "balcony",
                "room_type": "balcony",
                "area": 3.1,
                "floor_polygon": [
                    {"x": 12.0, "y": 15.4},
                    {"x": 12.0, "y": 14.3},
                    {"x": 9.0, "y": 14.3},
                    {"x": 9.0, "y": 15.4}
                ]
            },
            {
                "id": "interior_door|0",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 7.4, "y": 8.5},
                    {"x": 7.4, "y": 8.4},
                    {"x": 5.8, "y": 8.4},
                    {"x": 5.8, "y": 8.5}
                ]
            },
            {
                "id": "interior_door|1",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.8, "y": 7.4},
                    {"x": 8.3, "y": 7.4},
                    {"x": 8.3, "y": 7.5},
                    {"x": 8.8, "y": 7.5}
                ]
            },
            {
                "id": "interior_door|2",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.3, "y": 8.4},
                    {"x": 8.3, "y": 8.5},
                    {"x": 8.8, "y": 8.5},
                    {"x": 8.8, "y": 8.4}
                ]
            },
            {
                "id": "interior_door|3",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 10.8, "y": 5.9},
                    {"x": 10.8, "y": 5.8},
                    {"x": 9.2, "y": 5.8},
                    {"x": 9.2, "y": 5.9}
                ]
            },
            {
                "id": "interior_door|4",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 12.1, "y": 4.6},
                    {"x": 11.3, "y": 4.6},
                    {"x": 11.3, "y": 4.7},
                    {"x": 12.1, "y": 4.7}
                ]
            },
            {
                "id": "interior_door|5",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 11.2, "y": 14.0},
                    {"x": 9.7, "y": 14.0},
                    {"x": 9.7, "y": 14.1},
                    {"x": 11.2, "y": 14.1}
                ]
            },
            {
                "id": "front_door",
                "room_type": "front_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 12.2, "y": 5.3},
                    {"x": 12.1, "y": 5.3},
                    {"x": 12.1, "y": 5.8},
                    {"x": 12.2, "y": 5.8}
                ]
            }
        ]
    }

@pytest.fixture
def complex_ds2d_data():
    """Fixture providing complex DS2D data with 8 spaces for connectivity testing"""
    return {
        "room_count": 8,
        "spaces": [
            {
                "id": "storage",
                "room_type": "storage",
                "area": 2.2,
                "floor_polygon": [
                    {"x": 4.3, "y": 14.3},
                    {"x": 5.6, "y": 14.3},
                    {"x": 5.6, "y": 12.7},
                    {"x": 4.3, "y": 12.7}
                ]
            },
            {
                "id": "living_room",
                "room_type": "living_room",
                "area": 35.8,
                "floor_polygon": [
                    {"x": 8.6, "y": 14.3},
                    {"x": 12.5, "y": 14.3},
                    {"x": 12.5, "y": 12},
                    {"x": 14.3, "y": 12},
                    {"x": 14.3, "y": 7.7},
                    {"x": 8.3, "y": 7.7},
                    {"x": 8.3, "y": 6.6},
                    {"x": 7.4, "y": 6.6},
                    {"x": 7.4, "y": 8.9},
                    {"x": 8.6, "y": 8.9}
                ]
            },
            {
                "id": "bedroom|0",
                "room_type": "bedroom",
                "area": 7.1,
                "floor_polygon": [
                    {"x": 8.3, "y": 9.1},
                    {"x": 7.4, "y": 9.1},
                    {"x": 7.4, "y": 9.6},
                    {"x": 5.9, "y": 9.6},
                    {"x": 5.9, "y": 12.4},
                    {"x": 8.3, "y": 12.4}
                ]
            },
            {
                "id": "kitchen",
                "room_type": "kitchen",
                "area": 3.9,
                "floor_polygon": [
                    {"x": 5.9, "y": 14.3},
                    {"x": 8.3, "y": 14.3},
                    {"x": 8.3, "y": 12.7},
                    {"x": 5.9, "y": 12.7}
                ]
            },
            {
                "id": "bathroom|0",
                "room_type": "bathroom",
                "area": 3.3,
                "floor_polygon": [
                    {"x": 5, "y": 9.3},
                    {"x": 7.1, "y": 9.3},
                    {"x": 7.1, "y": 7.7},
                    {"x": 5, "y": 7.7}
                ]
            },
            {
                "id": "bedroom|1",
                "room_type": "bedroom",
                "area": 13.3,
                "floor_polygon": [
                    {"x": 8.6, "y": 7.5},
                    {"x": 13.8, "y": 7.5},
                    {"x": 13.8, "y": 6.2},
                    {"x": 14.3, "y": 6.2},
                    {"x": 14.3, "y": 4.4},
                    {"x": 10.1, "y": 4.4},
                    {"x": 10.1, "y": 6.6},
                    {"x": 8.6, "y": 6.6}
                ]
            },
            {
                "id": "bedroom|2",
                "room_type": "bedroom",
                "area": 10.7,
                "floor_polygon": [
                    {"x": 7.7, "y": 4.4},
                    {"x": 5.4, "y": 4.4},
                    {"x": 5.4, "y": 3.7},
                    {"x": 3.7, "y": 3.7},
                    {"x": 3.7, "y": 4.4},
                    {"x": 4.4, "y": 4.4},
                    {"x": 4.4, "y": 7.5},
                    {"x": 7.1, "y": 7.5},
                    {"x": 7.1, "y": 6.3},
                    {"x": 7.7, "y": 6.3}
                ]
            },
            {
                "id": "bathroom|1",
                "room_type": "bathroom",
                "area": 3.5,
                "floor_polygon": [
                    {"x": 8, "y": 4.4},
                    {"x": 8, "y": 6.3},
                    {"x": 9.8, "y": 6.3},
                    {"x": 9.8, "y": 4.4}
                ]
            },
            {
                "id": "interior_door|0",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 7.6, "y": 9.1},
                    {"x": 8.4, "y": 9.1},
                    {"x": 8.4, "y": 8.9},
                    {"x": 7.6, "y": 8.9}
                ]
            },
            {
                "id": "interior_door|1",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.5, "y": 12.9},
                    {"x": 8.4, "y": 12.9},
                    {"x": 8.4, "y": 13.5},
                    {"x": 8.5, "y": 13.5}
                ]
            },
            {
                "id": "interior_door|2",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 5.7, "y": 12.9},
                    {"x": 5.7, "y": 13.4},
                    {"x": 5.8, "y": 13.4},
                    {"x": 5.8, "y": 12.9}
                ]
            },
            {
                "id": "interior_door|3",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 7.2, "y": 7.9},
                    {"x": 7.2, "y": 8.5},
                    {"x": 7.3, "y": 8.5},
                    {"x": 7.3, "y": 7.9}
                ]
            },
            {
                "id": "interior_door|4",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.4, "y": 7.5},
                    {"x": 8.5, "y": 7.5},
                    {"x": 8.5, "y": 6.8},
                    {"x": 8.4, "y": 6.8}
                ]
            },
            {
                "id": "interior_door|5",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 7.3, "y": 6.5},
                    {"x": 7.2, "y": 6.5},
                    {"x": 7.2, "y": 7.3},
                    {"x": 7.3, "y": 7.3}
                ]
            },
            {
                "id": "interior_door|6",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.9, "y": 6.4},
                    {"x": 9.1, "y": 6.4},
                    {"x": 9.1, "y": 6.5},
                    {"x": 9.9, "y": 6.5}
                ]
            },
            {
                "id": "front_door",
                "room_type": "front_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 12.5, "y": 14.3},
                    {"x": 11.7, "y": 14.3},
                    {"x": 11.7, "y": 14.5},
                    {"x": 12.5, "y": 14.5}
                ]
            }
        ]
    }

@pytest.fixture
def generated_ds2d_data():
    """Fixture providing generated DS2D data with disconnected spaces for testing"""
    return {
        "room_count": 7,
        "spaces": [
            {
                "id": "bedroom|0",
                "room_type": "bedroom",
                "area": 14.7,
                "floor_polygon": [
                    {"x": 4.9, "y": 13.9},
                    {"x": 8.7, "y": 13.9},
                    {"x": 8.7, "y": 10.0},
                    {"x": 4.9, "y": 10.0}
                ]
            },
            {
                "id": "balcony|0",
                "room_type": "balcony",
                "area": 3.1,
                "floor_polygon": [
                    {"x": 8.9, "y": 14.2},
                    {"x": 8.9, "y": 15.0},
                    {"x": 12.7, "y": 15.0},
                    {"x": 12.7, "y": 14.2}
                ]
            },
            {
                "id": "bedroom|1",
                "room_type": "bedroom",
                "area": 12.2,
                "floor_polygon": [
                    {"x": 4.9, "y": 6.5},
                    {"x": 8.7, "y": 6.5},
                    {"x": 8.7, "y": 3.3},
                    {"x": 4.9, "y": 3.3}
                ]
            },
            {
                "id": "balcony|1",
                "room_type": "balcony",
                "area": 1.1,
                "floor_polygon": [
                    {"x": 3.8, "y": 3.3},
                    {"x": 3.8, "y": 4.4},
                    {"x": 4.7, "y": 4.4},
                    {"x": 4.7, "y": 3.3}
                ]
            },
            {
                "id": "bathroom",
                "room_type": "bathroom",
                "area": 8.4,
                "floor_polygon": [
                    {"x": 4.9, "y": 9.7},
                    {"x": 7.5, "y": 9.7},
                    {"x": 7.5, "y": 6.8},
                    {"x": 4.9, "y": 6.8}
                ]
            },
            {
                "id": "kitchen",
                "room_type": "kitchen",
                "area": 6.4,
                "floor_polygon": [
                    {"x": 11.9, "y": 3.3},
                    {"x": 9.0, "y": 3.3},
                    {"x": 9.0, "y": 5.7},
                    {"x": 11.9, "y": 5.7}
                ]
            },
            {
                "id": "living_room",
                "room_type": "living_room",
                "area": 29.3,
                "floor_polygon": [
                    {"x": 7.7, "y": 6.8},
                    {"x": 7.7, "y": 9.7},
                    {"x": 8.9, "y": 9.7},
                    {"x": 8.9, "y": 13.9},
                    {"x": 12.7, "y": 13.9},
                    {"x": 12.7, "y": 7.5},
                    {"x": 11.9, "y": 7.5},
                    {"x": 11.9, "y": 5.9},
                    {"x": 8.9, "y": 5.9},
                    {"x": 8.9, "y": 6.8}
                ]
            },
            {
                "id": "interior_door|0",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 4.8, "y": 3.2},
                    {"x": 4.8, "y": 3.0},
                    {"x": 3.7, "y": 3.0},
                    {"x": 3.7, "y": 3.2}
                ]
            },
            {
                "id": "interior_door|1",
                "room_type": "interior_door",
                "area": 0.4,
                "floor_polygon": [
                    {"x": 9.1, "y": 14.0},
                    {"x": 9.1, "y": 14.1},
                    {"x": 12.2, "y": 14.1},
                    {"x": 12.2, "y": 14.0}
                ]
            },
            {
                "id": "interior_door|2",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.8, "y": 10.0},
                    {"x": 8.6, "y": 10.0},
                    {"x": 8.6, "y": 10.4},
                    {"x": 8.8, "y": 10.4}
                ]
            },
            {
                "id": "interior_door|3",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.6, "y": 6.6},
                    {"x": 8.6, "y": 7.0},
                    {"x": 8.8, "y": 7.0},
                    {"x": 8.8, "y": 6.6}
                ]
            },
            {
                "id": "interior_door|4",
                "room_type": "interior_door",
                "area": 0.0,
                "floor_polygon": [
                    {"x": 7.5, "y": 8.2},
                    {"x": 7.7, "y": 8.2},
                    {"x": 7.7, "y": 7.9},
                    {"x": 7.5, "y": 7.9}
                ]
            },
            {
                "id": "interior_door|5",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 9.1, "y": 5.8},
                    {"x": 10.6, "y": 5.8},
                    {"x": 10.6, "y": 5.7},
                    {"x": 9.1, "y": 5.7}
                ]
            },
            {
                "id": "front_door",
                "room_type": "front_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 12.7, "y": 7.5},
                    {"x": 12.7, "y": 8.2},
                    {"x": 12.9, "y": 8.2},
                    {"x": 12.9, "y": 7.5}
                ]
            }
        ]
    }

@pytest.fixture
def double_connection_balcony_ds2d_data():
    """8-room floorplan with balcony having double connections to bedroom and study room"""
    return {
        "room_count": 8,
        "spaces": [
            {
                "id": "balcony|0",
                "room_type": "balcony",
                "area": 8.5,
                "floor_polygon": [
                    {"x": 13.4, "y": 3},
                    {"x": 6.6, "y": 3},
                    {"x": 6.6, "y": 4.3},
                    {"x": 13.4, "y": 4.3}
                ]
            },
            {
                "id": "bedroom|0",
                "room_type": "bedroom",
                "area": 11.4,
                "floor_polygon": [
                    {"x": 9.9, "y": 4.6},
                    {"x": 9.9, "y": 7.9},
                    {"x": 13.4, "y": 7.9},
                    {"x": 13.4, "y": 4.6}
                ]
            },
            {
                "id": "bathroom",
                "room_type": "bathroom",
                "area": 4.2,
                "floor_polygon": [
                    {"x": 11.2, "y": 8.2},
                    {"x": 11.2, "y": 10.1},
                    {"x": 13.4, "y": 10.1},
                    {"x": 13.4, "y": 8.2}
                ]
            },
            {
                "id": "bedroom|1",
                "room_type": "bedroom",
                "area": 11.5,
                "floor_polygon": [
                    {"x": 9.9, "y": 9.6},
                    {"x": 9.9, "y": 13.5},
                    {"x": 13.4, "y": 13.5},
                    {"x": 13.4, "y": 10.4},
                    {"x": 11, "y": 10.4},
                    {"x": 11, "y": 9.6}
                ]
            },
            {
                "id": "balcony|1",
                "room_type": "balcony",
                "area": 4.1,
                "floor_polygon": [
                    {"x": 9.9, "y": 13.8},
                    {"x": 9.9, "y": 15},
                    {"x": 13.4, "y": 15},
                    {"x": 13.4, "y": 13.8}
                ]
            },
            {
                "id": "living_room",
                "room_type": "living_room",
                "area": 27.9,
                "floor_polygon": [
                    {"x": 11, "y": 9.3},
                    {"x": 11, "y": 8.2},
                    {"x": 5.4, "y": 8.2},
                    {"x": 5.4, "y": 14.4},
                    {"x": 9.6, "y": 14.4},
                    {"x": 9.6, "y": 9.3}
                ]
            },
            {
                "id": "kitchen",
                "room_type": "kitchen",
                "area": 5.6,
                "floor_polygon": [
                    {"x": 4.6, "y": 4.6},
                    {"x": 4.6, "y": 7.9},
                    {"x": 6.3, "y": 7.9},
                    {"x": 6.3, "y": 4.6}
                ]
            },
            {
                "id": "study_room",
                "room_type": "study_room",
                "area": 10,
                "floor_polygon": [
                    {"x": 9.6, "y": 4.6},
                    {"x": 6.6, "y": 4.6},
                    {"x": 6.6, "y": 7.9},
                    {"x": 9.6, "y": 7.9}
                ]
            },
            {
                "id": "interior_door|0",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 12.5, "y": 4.5},
                    {"x": 12.5, "y": 4.4},
                    {"x": 10.8, "y": 4.4},
                    {"x": 10.8, "y": 4.5}
                ]
            },
            {
                "id": "interior_door|1",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 12.5, "y": 13.7},
                    {"x": 12.5, "y": 13.6},
                    {"x": 10.8, "y": 13.6},
                    {"x": 10.8, "y": 13.7}
                ]
            },
            {
                "id": "interior_door|2",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 10.1, "y": 7.9},
                    {"x": 10.1, "y": 8.1},
                    {"x": 10.8, "y": 8.1},
                    {"x": 10.8, "y": 7.9}
                ]
            },
            {
                "id": "interior_door|3",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 11, "y": 9.1},
                    {"x": 11.2, "y": 9.1},
                    {"x": 11.2, "y": 8.4},
                    {"x": 11, "y": 8.4}
                ]
            },
            {
                "id": "interior_door|4",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 10.8, "y": 9.4},
                    {"x": 9.8, "y": 9.4},
                    {"x": 9.8, "y": 9.5},
                    {"x": 10.8, "y": 9.5}
                ]
            },
            {
                "id": "interior_door|5",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 6.2, "y": 7.9},
                    {"x": 5.3, "y": 7.9},
                    {"x": 5.3, "y": 8.1},
                    {"x": 6.2, "y": 8.1}
                ]
            },
            {
                "id": "interior_door|6",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 7.3, "y": 4.4},
                    {"x": 7.3, "y": 4.5},
                    {"x": 9, "y": 4.5},
                    {"x": 9, "y": 4.4}
                ]
            },
            {
                "id": "interior_door|7",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.5, "y": 8.1},
                    {"x": 9.5, "y": 7.9},
                    {"x": 8.6, "y": 7.9},
                    {"x": 8.6, "y": 8.1}
                ]
            },
            {
                "id": "front_door",
                "room_type": "front_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 5.3, "y": 9.2},
                    {"x": 5.2, "y": 9.2},
                    {"x": 5.2, "y": 10.3},
                    {"x": 5.3, "y": 10.3}
                ]
            }
        ]
    }

@pytest.fixture
def containment_issue_ds2d_data():
    """8-room floorplan with bathroom contained inside bedroom (invalid case)"""
    return {
        "room_count": 8,
        "spaces": [
            {
                "id": "kitchen",
                "room_type": "kitchen",
                "area": 5.2,
                "floor_polygon": [
                    {"x": 8.2, "y": 3.2},
                    {"x": 5.3, "y": 3.2},
                    {"x": 5.3, "y": 5.0},
                    {"x": 8.2, "y": 5.0}
                ]
            },
            {
                "id": "storage",
                "room_type": "storage",
                "area": 1.9,
                "floor_polygon": [
                    {"x": 9.6, "y": 5.0},
                    {"x": 9.6, "y": 3.2},
                    {"x": 8.5, "y": 3.2},
                    {"x": 8.5, "y": 5.0}
                ]
            },
            {
                "id": "balcony",
                "room_type": "balcony",
                "area": 5.1,
                "floor_polygon": [
                    {"x": 5.3, "y": 13.4},
                    {"x": 5.3, "y": 14.8},
                    {"x": 9.1, "y": 14.8},
                    {"x": 9.1, "y": 13.4}
                ]
            },
            {
                "id": "bathroom|0",
                "room_type": "bathroom",
                "area": 3.2,
                "floor_polygon": [
                    {"x": 11.9, "y": 9.6},
                    {"x": 11.9, "y": 8.2},
                    {"x": 9.7, "y": 8.2},
                    {"x": 9.7, "y": 9.6}
                ]
            },
            {
                "id": "bedroom|0",
                "room_type": "bedroom",
                "area": 12.1,
                "floor_polygon": [
                    {"x": 9.4, "y": 9.6},
                    {"x": 9.4, "y": 9.9},
                    {"x": 12.9, "y": 9.9},
                    {"x": 12.9, "y": 6.7},
                    {"x": 9.9, "y": 6.7},
                    {"x": 9.9, "y": 6.0},
                    {"x": 9.2, "y": 6.0},
                    {"x": 9.2, "y": 9.6}
                ]
            },
            {
                "id": "bedroom|1",
                "room_type": "bedroom",
                "area": 12.3,
                "floor_polygon": [
                    {"x": 10.3, "y": 14.1},
                    {"x": 12.9, "y": 14.1},
                    {"x": 12.9, "y": 10.2},
                    {"x": 9.4, "y": 10.2},
                    {"x": 9.4, "y": 13.1},
                    {"x": 10.3, "y": 13.1}
                ]
            },
            {
                "id": "bathroom|1",
                "room_type": "bathroom",
                "area": 4.3,
                "floor_polygon": [
                    {"x": 6.7, "y": 6.0},
                    {"x": 6.7, "y": 8.5},
                    {"x": 8.4, "y": 8.5},
                    {"x": 8.4, "y": 6.0}
                ]
            },
            {
                "id": "living_room",
                "room_type": "living_room",
                "area": 28.2,
                "floor_polygon": [
                    {"x": 8.9, "y": 13.1},
                    {"x": 8.9, "y": 9.9},
                    {"x": 8.9, "y": 9.9},
                    {"x": 8.9, "y": 8.8},
                    {"x": 8.7, "y": 8.8},
                    {"x": 8.7, "y": 8.9},
                    {"x": 6.4, "y": 8.9},
                    {"x": 6.4, "y": 5.3},
                    {"x": 5.1, "y": 5.3},
                    {"x": 5.1, "y": 13.1}
                ]
            },
            {
                "id": "interior_door|0",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.5, "y": 9.1},
                    {"x": 9.5, "y": 9.7},
                    {"x": 9.6, "y": 9.7},
                    {"x": 9.6, "y": 9.1}
                ]
            },
            {
                "id": "interior_door|1",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.4, "y": 5.2},
                    {"x": 9.4, "y": 5.1},
                    {"x": 8.7, "y": 5.1},
                    {"x": 8.7, "y": 5.2}
                ]
            },
            {
                "id": "interior_door|2",
                "room_type": "interior_door",
                "area": 0.2,
                "floor_polygon": [
                    {"x": 6.4, "y": 5.2},
                    {"x": 6.4, "y": 5.1},
                    {"x": 5.3, "y": 5.1},
                    {"x": 5.3, "y": 5.2}
                ]
            },
            {
                "id": "interior_door|3",
                "room_type": "interior_door",
                "area": 0.4,
                "floor_polygon": [
                    {"x": 8.4, "y": 13.4},
                    {"x": 8.4, "y": 13.2},
                    {"x": 5.8, "y": 13.2},
                    {"x": 5.8, "y": 13.4}
                ]
            },
            {
                "id": "interior_door|4",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.1, "y": 9.8},
                    {"x": 9.1, "y": 9.1},
                    {"x": 9.0, "y": 9.1},
                    {"x": 9.0, "y": 9.8}
                ]
            },
            {
                "id": "interior_door|5",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.3, "y": 10.3},
                    {"x": 9.1, "y": 10.3},
                    {"x": 9.1, "y": 10.9},
                    {"x": 9.3, "y": 10.9}
                ]
            },
            {
                "id": "interior_door|6",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.6, "y": 7.9},
                    {"x": 8.5, "y": 7.9},
                    {"x": 8.5, "y": 8.5},
                    {"x": 8.6, "y": 8.5}
                ]
            },
            {
                "id": "front_door",
                "room_type": "front_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 6.3, "y": 5.2},
                    {"x": 6.3, "y": 5.1},
                    {"x": 5.5, "y": 5.1},
                    {"x": 5.5, "y": 5.2}
                ]
            }
        ]
    }

@pytest.fixture
def multiple_doors_ds2d_data():
    """Fixture with multiple doors connecting the same rooms (invalid case)"""
    return {
        "room_count": 5,
        "spaces": [
            {
                "id": "bedroom",
                "room_type": "bedroom",
                "area": 19.9,
                "floor_polygon": [
                    {"x": 13.0, "y": 8.2},
                    {"x": 13.0, "y": 3.7},
                    {"x": 8.6, "y": 3.7},
                    {"x": 8.6, "y": 8.2}
                ]
            },
            {
                "id": "balcony",
                "room_type": "balcony",
                "area": 5.3,
                "floor_polygon": [
                    {"x": 6.1, "y": 5.4},
                    {"x": 6.1, "y": 7.9},
                    {"x": 8.3, "y": 7.9},
                    {"x": 8.3, "y": 5.4}
                ]
            },
            {
                "id": "bathroom",
                "room_type": "bathroom",
                "area": 6.2,
                "floor_polygon": [
                    {"x": 13.0, "y": 8.4},
                    {"x": 10.1, "y": 8.4},
                    {"x": 10.1, "y": 10.7},
                    {"x": 13.0, "y": 10.7}
                ]
            },
            {
                "id": "kitchen",
                "room_type": "kitchen",
                "area": 7.8,
                "floor_polygon": [
                    {"x": 13.0, "y": 11.0},
                    {"x": 10.1, "y": 11.0},
                    {"x": 10.1, "y": 12.0},
                    {"x": 11.0, "y": 12.0},
                    {"x": 11.0, "y": 14.3},
                    {"x": 13.0, "y": 14.3}
                ]
            },
            {
                "id": "living_room",
                "room_type": "living_room",
                "area": 18.0,
                "floor_polygon": [
                    {"x": 5.8, "y": 8.2},
                    {"x": 5.8, "y": 8.2},
                    {"x": 5.8, "y": 14.3},
                    {"x": 10.8, "y": 14.3},
                    {"x": 10.8, "y": 12.3},
                    {"x": 9.8, "y": 12.3},
                    {"x": 9.8, "y": 8.4},
                    {"x": 8.3, "y": 8.4},
                    {"x": 8.3, "y": 8.2},
                    {"x": 5.8, "y": 8.2}
                ]
            },
            {
                "id": "interior_door|0",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 6.0, "y": 8.2},
                    {"x": 6.8, "y": 8.2},
                    {"x": 6.8, "y": 8.0},
                    {"x": 6.0, "y": 8.0}
                ]
            },
            {
                "id": "interior_door|1",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 8.8, "y": 8.2},
                    {"x": 8.8, "y": 8.4},
                    {"x": 9.6, "y": 8.4},
                    {"x": 9.6, "y": 8.2}
                ]
            },
            {
                "id": "interior_door|2",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 9.9, "y": 9.7},
                    {"x": 10.1, "y": 9.7},
                    {"x": 10.1, "y": 8.7},
                    {"x": 9.9, "y": 8.7}
                ]
            },
            {
                "id": "interior_door|3",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 10.0, "y": 11.2},
                    {"x": 9.8, "y": 11.2},
                    {"x": 9.8, "y": 12.0},
                    {"x": 10.0, "y": 12.0}
                ]
            },
            {
                "id": "interior_door|4",
                "room_type": "interior_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 10.0, "y": 12.1},
                    {"x": 10.0, "y": 12.2},
                    {"x": 10.8, "y": 12.2},
                    {"x": 10.8, "y": 12.1}
                ]
            },
            {
                "id": "front_door",
                "room_type": "front_door",
                "area": 0.1,
                "floor_polygon": [
                    {"x": 5.7, "y": 14.5},
                    {"x": 6.7, "y": 14.5},
                    {"x": 6.7, "y": 14.3},
                    {"x": 5.7, "y": 14.3}
                ]
            }
        ]
    }


class TestConnectivity:
    """Test connectivity"""
    
    def test_sample_ds2d_connectivity(self, sample_ds2d_data):
        """Test expected connectivity of the 7-room sample"""
        graph = RPLANGraph.from_ds2d(sample_ds2d_data)
        
        # Should have 7 room nodes (excluding doors)
        assert len(graph.graph.nodes()) == 7
        
        # Based on door positions and validation, expect 5 connections (front door excluded)
        assert len(graph.graph.edges()) == 5
        
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

    def test_complex_floorplan_connectivity(self, complex_ds2d_data):
        """Test connectivity of complex 8-room floorplan with expected adjacency"""
        graph = RPLANGraph.from_ds2d(complex_ds2d_data)
        
        # Should have 8 room nodes (excluding doors)
        assert len(graph.graph.nodes()) == 8
        
        # Should have exactly 7 edges for this floorplan (front door excluded)
        assert len(graph.graph.edges()) == 7
        
        # Should be fully connected
        is_connected = nx.is_connected(graph.graph)
        assert is_connected, "Complex floorplan should be fully connected"
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity based on architectural layout (verified to be 100% accurate)
        expected_adjacency = {
            'storage': ['kitchen'], 
            'living_room': ['bedroom|0', 'kitchen', 'bathroom|0', 'bedroom|1', 'bedroom|2'], 
            'bedroom|0': ['living_room'], 
            'kitchen': ['living_room', 'storage'], 
            'bathroom|0': ['living_room'], 
            'bedroom|1': ['living_room', 'bathroom|1'], 
            'bedroom|2': ['living_room'], 
            'bathroom|1': ['bedroom|1']
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
        
        # Verify living_room is the central hub with 5 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 5, f"Living room should have 5 connections, got {living_room_connections}"

    def test_generated_floorplan_connectivity(self, generated_ds2d_data):
        """Test connectivity of generated floorplan with expected disconnected spaces"""
        graph = RPLANGraph.from_ds2d(generated_ds2d_data)
        
        # Should have 7 room nodes (excluding doors)
        assert len(graph.graph.nodes()) == 7
        
        # This floorplan has disconnected spaces, so not fully connected
        is_connected = nx.is_connected(graph.graph)
        assert not is_connected, "Generated floorplan should NOT be fully connected (has isolated spaces)"
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - many spaces are isolated (front door excluded)
        expected_adjacency = {
            'bedroom|0': [],
            'balcony|0': ['living_room'],
            'bedroom|1': [],
            'balcony|1': [],
            'bathroom': ['living_room'],
            'kitchen': ['living_room'],
            'living_room': ['balcony|0', 'bathroom', 'kitchen']
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
        
        # Verify living_room has exactly 3 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 3, f"Living room should have 3 connections, got {living_room_connections}"
        
        # Check connected components
        components = list(nx.connected_components(graph.graph))
        assert len(components) == 4, f"Should have 4 connected components, got {len(components)}"
        
        # Find the largest component (should contain living_room)
        largest_component = max(components, key=len)
        assert len(largest_component) == 4, f"Largest component should have 4 spaces, got {len(largest_component)}"

    def test_double_connection_balcony_connectivity(self, double_connection_balcony_ds2d_data):
        """Test connectivity of 8-room floorplan with balcony having double connections"""
        graph = RPLANGraph.from_ds2d(double_connection_balcony_ds2d_data)
        
        # Should have 8 room nodes (excluding doors)
        assert len(graph.graph.nodes()) == 8
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - balcony|0 has double connections (to bedroom|0 and study_room)
        expected_adjacency = {
            'balcony|0': ['bedroom|0', 'study_room'],  # Double connection case
            'bedroom|0': ['balcony|0', 'living_room'], 
            'bathroom': ['living_room'], 
            'bedroom|1': ['balcony|1', 'living_room'], 
            'balcony|1': ['bedroom|1'], 
            'living_room': ['bedroom|0', 'bathroom', 'bedroom|1', 'kitchen', 'study_room'], 
            'kitchen': ['living_room'], 
            'study_room': ['balcony|0', 'living_room']
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
        
        # Verify living_room is the central hub with 5 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 5, f"Living room should have 5 connections, got {living_room_connections}"
        
        # Check that the graph is fully connected
        is_connected = nx.is_connected(graph.graph)
        assert is_connected, "New floorplan should be fully connected"
        
        # Verify total number of edges matches expected connections
        total_edges = len(graph.graph.edges())
        expected_edges = sum(len(neighbors) for neighbors in expected_adjacency.values()) // 2
        assert total_edges == expected_edges, f"Expected {expected_edges} edges, got {total_edges}"

    def test_containment_issue_connectivity(self, containment_issue_ds2d_data):
        """Test that spaces contained within other spaces do not create invalid connections"""
        graph = RPLANGraph.from_ds2d(containment_issue_ds2d_data)
        
        # Should have 8 room nodes (excluding doors)
        assert len(graph.graph.nodes()) == 8
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - bathroom|0 should NOT connect to bedroom|0 (containment case)
        expected_adjacency = {
            'kitchen': ['living_room'], 
            'storage': [], 
            'balcony': ['living_room'], 
            'bathroom|0': [],  # Should be empty (not connected to bedroom|0 due to containment)
            'bedroom|0': ['living_room'],  # Should only connect to living_room
            'bedroom|1': ['living_room'], 
            'bathroom|1': [], 
            'living_room': ['kitchen', 'balcony', 'bedroom|0', 'bedroom|1']
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
        
        # Verify living_room is the central hub with 4 connections
        living_room_connections = len(labeled_adj.get('living_room', []))
        assert living_room_connections == 4, f"Living room should have 4 connections, got {living_room_connections}"
        
        # Verify total number of edges matches expected connections
        total_edges = len(graph.graph.edges())
        expected_edges = sum(len(neighbors) for neighbors in expected_adjacency.values()) // 2
        assert total_edges == expected_edges, f"Expected {expected_edges} edges, got {total_edges}"

    def test_front_door_exclusion(self):
        """Test that front doors do not create connections between spaces"""
        # Test data with a front door between two spaces
        test_data = {
            "room_count": 3,
            "spaces": [
                {
                    "id": "room_a",
                    "room_type": "living_room",
                    "area": 10.0,
                    "floor_polygon": [
                        {"x": 0.0, "y": 0.0},
                        {"x": 4.0, "y": 0.0},
                        {"x": 4.0, "y": 4.0},
                        {"x": 0.0, "y": 4.0}
                    ]
                },
                {
                    "id": "room_b",
                    "room_type": "kitchen",
                    "area": 10.0,
                    "floor_polygon": [
                        {"x": 4.0, "y": 0.0},
                        {"x": 8.0, "y": 0.0},
                        {"x": 8.0, "y": 4.0},
                        {"x": 4.0, "y": 4.0}
                    ]
                },
                {
                    "id": "front_door",
                    "room_type": "front_door",
                    "area": 0.1,
                    "floor_polygon": [
                        {"x": 3.8, "y": 2.0},
                        {"x": 4.2, "y": 2.0},
                        {"x": 4.2, "y": 2.5},
                        {"x": 3.8, "y": 2.5}
                    ]
                }
            ]
        }
        
        graph = RPLANGraph.from_ds2d(test_data)
        
        # Should have 2 room nodes
        assert len(graph.graph.nodes()) == 2
        
        # Should have NO edges (front door excluded)
        assert len(graph.graph.edges()) == 0, "Front door should not create connections between spaces"
        
        # Verify spaces are not connected
        labeled_adj = graph.to_labeled_adjacency()
        assert labeled_adj['living_room'] == []
        assert labeled_adj['kitchen'] == []

    def test_single_door_connectivity(self, multiple_doors_ds2d_data):
        """Test that rooms with multiple doors are not connected (invalid connections rejected)"""
        graph = RPLANGraph.from_ds2d(multiple_doors_ds2d_data)
        
        # Should have 5 room nodes (excluding doors)
        assert len(graph.graph.nodes()) == 5
        
        # Convert to labeled adjacency to check expected connections
        labeled_adj = graph.to_labeled_adjacency()
        
        # Expected connectivity - rooms with multiple doors should NOT be connected
        # The kitchen and living_room have multiple doors, so they should not be connected
        expected_adjacency = {
            'bedroom': ['living_room'],
            'balcony': ['living_room'],
            'bathroom': ['living_room'],
            'kitchen': [],  # No connection due to multiple doors
            'living_room': ['bedroom', 'balcony', 'bathroom']  # No kitchen due to multiple doors
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
        expected_edges = 3  # 3 unique connections: bedroom-living_room, balcony-living_room, bathroom-living_room
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
        assert len(living_room_component) == 4, f"Living room component should have 4 rooms, got {len(living_room_component)}"
        assert len(kitchen_component) == 1, f"Kitchen component should have 1 room, got {len(kitchen_component)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 