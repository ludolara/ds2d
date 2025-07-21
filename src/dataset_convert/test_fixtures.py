import pytest

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

@pytest.fixture
def front_door_exclusion_data():
    return {
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
