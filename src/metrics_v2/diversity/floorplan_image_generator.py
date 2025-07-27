import json
import numpy as np
import drawSvg as drawsvg
import cairosvg
import PIL.Image as Image
from PIL import ImageDraw
import io
import cv2 as cv

class HouseDiffusionVisualizerDS2D:
    """
    Enhanced HouseDiffusion visualizer that works with the custom my_data_format.json format.
    Provides the same visualization as HouseDiffusionVisualizer but with different input format.
    """
    
    def __init__(self, resolution=256):
        """
        Initialize the DS2D visualizer.
        
        Args:
            resolution (int): The output image resolution (default: 256)
        """
        self.resolution = resolution
        
        # Room type to ID mapping for consistency with original system
        self.ROOM_TYPE_TO_ID = {
            "living_room": 1, 
            "kitchen": 2, 
            "bedroom": 3, 
            "bathroom": 4, 
            "balcony": 5, 
            "entrance": 6, 
            "dining_room": 7, 
            "study_room": 8,
            "storage": 9, 
            # "front_door": 10, 
            "unknown": 16, 
            "front_door": 15,
            "interior_door": 17,
        }
              
        
        # Exact room type color mapping from the original HouseDiffusion system
        self.ID_COLOR = {
            1: '#EE4D4D',    # Living room
            2: '#C67C7B',    # Bedroom  
            3: '#FFD274',    # Kitchen
            4: '#BEBEBE',    # Bathroom
            5: '#BFE3E8',    # Balcony
            6: '#7BA779',    # Corridor
            7: '#E87A90',    # Dining room
            8: '#FF8C69',    # Study
            9: '#1F849B',    # Storage
            15: '#727171',   
            17: '#D3A2C7'    
        }
        
        # Door indices from the original system (including interior doors and walls)
        self.door_indices = [15, 17]
    
    def reader_ds2d(self, filename):
        """
        Read and process the DS2D format JSON file using mask-based approach to eliminate gaps.
        This replicates the exact mask processing logic from the original HouseDiffusion system.
        
        Args:
            filename (str): Path to the my_data_format.json file
            
        Returns:
            list: rooms_data - processed room data in internal format
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        rooms_data = []
        
        # Find the bounding box of all spaces to normalize coordinates
        all_x = []
        all_y = []
        
        for room in data['spaces']:
            polygon = room['floor_polygon']
            for point in polygon:
                all_x.append(point['x'])
                all_y.append(point['y'])
        
        if not all_x or not all_y:
            return []
        
        # Calculate bounding box
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        
        # Calculate scale and offset to fit in [0, 1] range first
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)  # Leave some margin, fit to [0,1] range
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Process each room using the EXACT mask-based approach from original
        for room in data['spaces']:
            room_type_str = room['room_type']
            room_id = self.ROOM_TYPE_TO_ID.get(room_type_str, 1)  # Default to living room if unknown
            
            # Use mask-based approach like the original (exact same as build_graph_and_masks)
            im_size = 256
            out_size = 64
            
            # Create room mask (exact same as original)
            room_img = Image.new('L', (im_size, im_size))
            draw = ImageDraw.Draw(room_img)
            
            # Convert polygon coordinates to [0,1] then to image coordinates
            polygon_coords = []
            for point in room['floor_polygon']:
                # Normalize to [0, 1] range
                norm_x = (point['x'] - center_x) * scale + 0.5
                norm_y = (point['y'] - center_y) * scale + 0.5
                # Convert to image coordinates
                img_x = norm_x * im_size
                img_y = norm_y * im_size
                polygon_coords.append((img_x, img_y))
            
            # Draw the polygon to create a mask
            if len(polygon_coords) >= 3:
                draw.polygon(polygon_coords, fill='white')
            
            # Apply the exact same processing as original
            room_img = room_img.resize((out_size, out_size))
            room_array = np.array(room_img)
            
            # Find contours to get clean polygons (exact same as original)
            room_array = room_array.astype(np.uint8)
            room_array = cv.resize(room_array, (self.resolution, self.resolution), interpolation=cv.INTER_AREA)
            contours, _ = cv.findContours(room_array, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Take the largest contour
                largest_contour = max(contours, key=cv.contourArea)
                # Convert contour to polygon coordinates normalized to [-1, 1] like the model output
                polygon_coords = largest_contour[:, 0, :].astype(float) / self.resolution * 2 - 1
                rooms_data.append([polygon_coords, room_id])
        
        return rooms_data
    
    def visualize_floorplan_ds2d(self, filename, save_path=None, save_svg=False, show_edges=False):
        """
        Visualize floorplan from DS2D format JSON file.
        Uses the same visualization logic as the original HouseDiffusion system.
        
        Args:
            filename (str): Path to the my_data_format.json file
            save_path (str, optional): Path to save the output image
            save_svg (bool): Whether to save as SVG format as well
            show_edges (bool): Whether to show corner points/edges
            
        Returns:
            PIL.Image: The generated floorplan image
        """
        # Read and process the DS2D data
        spaces = self.reader_ds2d(filename)
        
        if not spaces:
            print("No spaces found in the data file!")
            return None
        
        # Create the visualization - EXACT same as save_samples function
        resolution = self.resolution
        
        # Initialize drawings (exact same as original)
        draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
        draw.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='black'))
        
        draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
        draw_color.append(drawsvg.Rectangle(0, 0, resolution, resolution, fill='white'))
        
        polys = []
        types = []
        
        # Process spaces to create polys and types (replicating the loop in save_samples)
        for room_coords, room_type in spaces:
            poly = []
            for coord in room_coords:
                # Convert from [-1, 1] to pixel coordinates (EXACT same as original)
                point = coord
                pred_center = False
                if pred_center:
                    point = point/2 + 1
                    point = point * resolution//2
                else:
                    point = point/2 + 0.5
                    point = point * resolution
                poly.append((point[0], point[1]))
            
            if len(poly) >= 3:
                polys.append(poly)
                types.append(room_type)
        
        # Draw spaces (non-door spaces first) - EXACT same as original
        for poly, c in zip(polys, types):
            if c in self.door_indices or c == 0:
                continue
            room_type = c
            if room_type in self.ID_COLOR:
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), 
                                              close=True, fill=self.ID_COLOR[room_type], 
                                              fill_opacity=1.0, stroke='black', stroke_width=1))
                
                # Add corner points if requested (exact same as original)
                if show_edges:
                    for corner in poly:
                        draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), 
                                                 fill=self.ID_COLOR[room_type], fill_opacity=1.0, 
                                                 stroke='gray', stroke_width=0.25))
        
        # Draw doors and walls - EXACT same as original (including interior doors)
        for poly, c in zip(polys, types):
            if c not in self.door_indices:
                continue
            room_type = c
            if room_type in self.ID_COLOR:
                # Use different stroke width for doors vs walls
                stroke_width = 2 if room_type in [11, 12] else 1  # Thicker for doors
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), 
                                              close=True, fill=self.ID_COLOR[room_type], 
                                              fill_opacity=1.0, stroke='black', stroke_width=stroke_width))
                
                # Add corner points if requested (exact same as original)
                if show_edges:
                    for corner in poly:
                        draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), 
                                                 fill=self.ID_COLOR[room_type], fill_opacity=1.0, 
                                                 stroke='gray', stroke_width=0.25))
        
        # Add summary statistics
        room_counts = {}
        for _, room_type in spaces:
            if room_type not in self.door_indices:
                room_name = {1: 'Living', 2: 'Bedroom', 3: 'Kitchen', 4: 'Bathroom', 5: 'Balcony',
                           6: 'Corridor', 7: 'Dining', 8: 'Study', 9: 'Storage'}.get(room_type, f'Type{room_type}')
                room_counts[room_name] = room_counts.get(room_name, 0) + 1
        
        door_counts = {}
        for _, room_type in spaces:
            if room_type in self.door_indices:
                door_name = {17: 'Interior Door', 15: 'Exterior Door'}.get(room_type, f'Type{room_type}')
                door_counts[door_name] = door_counts.get(door_name, 0) + 1
        
        # print(f"Rooms: {room_counts}")
        # print(f"Doors/Walls: {door_counts}")
        
        # Convert to image and save (enhanced with both SVG and PNG)
        if save_path:
            if save_svg:
                # Save SVG version
                svg_path = save_path.replace('.png', '.svg') if save_path.endswith('.png') else save_path
                draw_color.saveSvg(svg_path)
                # print(f"SVG saved: {svg_path}")
                
                # Also save PNG version
                png_path = save_path.replace('.svg', '.png') if save_path.endswith('.svg') else save_path
                svg_bytes = cairosvg.svg2png(draw_color.asSvg())
                img = Image.open(io.BytesIO(svg_bytes))
                img.save(png_path)
                # print(f"PNG saved: {png_path}")
            else:
                # Save PNG version
                svg_bytes = cairosvg.svg2png(draw_color.asSvg())
                img = Image.open(io.BytesIO(svg_bytes))
                img.save(save_path)
                # print(f"PNG saved: {save_path}")
        else:
            # Just convert for return
            svg_bytes = cairosvg.svg2png(draw_color.asSvg())
            img = Image.open(io.BytesIO(svg_bytes))
            
        return img


# Example usage and test
if __name__ == "__main__":
    import os
    
    # Create output directory
    os.makedirs('output_hd', exist_ok=True)
    
    # Example usage - DS2D format
    if os.path.exists('my_data_format.json'):
        print("Creating DS2D visualization...")
        visualizer_ds2d = HouseDiffusionVisualizerDS2D(resolution=256)
        img_ds2d = visualizer_ds2d.visualize_floorplan_ds2d(
            'my_data_format.json',
            save_path='output_hd/floorplan_ds2d_complete.png',
            save_svg=True,      # Save both SVG and PNG
            show_edges=True     # Show corner points
        )
        print("✅ DS2D visualization created!")
        print("📁 Output files:")
        print("   - PNG: output_hd/floorplan_ds2d_complete.png")
        print("   - SVG: output_hd/floorplan_ds2d_complete.svg")
        print("\n🏠 Includes: spaces and doors from my_data_format.json")
    else:
        print("❌ my_data_format.json not found!")
        print("Please make sure the file exists in the current directory.") 