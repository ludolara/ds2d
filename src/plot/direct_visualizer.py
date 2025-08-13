from typing import Dict, Optional
import matplotlib.pyplot as plt

class DirectVisualizer:
    """
    A simple visualizer for floorplans using matplotlib.
    """
    def __init__(
        self,
        room_colors: Optional[Dict[str, str]] = None,
        border_color: str = 'black',
        linewidth: float = 0.5,
        figsize: tuple = (10, 10),
        resolution: int = 256
    ):
        default_colors = {
            'living_room':   '#EE4D4D',
            'kitchen':       '#C67C7B',
            'bedroom':       '#FFD274',
            'bathroom':      '#BEBEBE',
            'balcony':       '#BFE3E8',
            'entrance':      '#7BA779',
            'dining_room':   '#E87A90',
            'study_room':    '#FF8C69',
            'storage':       '#1F849B',
            'front_door':    '#727171',
            'interior_door': '#D3A2C7',
            'unknown':       '#FFFFFF'
        }
        self.room_colors = room_colors or default_colors
        self.border_color = border_color
        self.linewidth = linewidth
        self.figsize = figsize
        self.resolution = resolution

    def plot(self, floorplan: Dict, save_path: Optional[str] = None, show: bool = True, dpi: int = 300) -> None:
        """
        Plot the floorplan.

        Args:
            floorplan: A dict containing 'spaces' (list of rooms) and optional 'doors'.
            save_path: Optional path to save the image. If None, image is not saved.
            show: Whether to show the plot. Default is True.
            dpi: DPI for saved image. Default is 300.
        """
        rooms = floorplan.get('spaces', [])

        fig, ax = plt.subplots(figsize=self.figsize)

        # Draw rooms
        for room in rooms:
            polygon = room.get('floor_polygon', [])
            if polygon:
                x_coords = [p['x'] for p in polygon]
                y_coords = [p['y'] for p in polygon]
                room_type = room.get('room_type', 'unknown')
                color = self.room_colors.get(room_type, self.room_colors['unknown'])
                ax.fill(x_coords, y_coords, color=color, edgecolor=self.border_color, 
                       linewidth=self.linewidth)

        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            
        if show:
            plt.show()
        else:
            plt.close()

    def save_visualization(self, floorplan: Dict, save_path: str, dpi: int = 150) -> bool:
        """
        Save a floorplan visualization to file without displaying it.
        
        Args:
            floorplan: A dict containing 'spaces' (list of rooms) and optional 'doors'.
            save_path: Path to save the image.
            dpi: DPI for saved image. Default is 150.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.plot(floorplan, save_path=save_path, show=False, dpi=dpi)
            return True
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False

    def generate_and_save_visualization(self, floorplan_data: Dict, save_path: str, dpi: int = None) -> bool:
        """
        Generate a direct visualization using matplotlib and save it at fixed resolution.
        
        Args:
            floorplan_data (dict): Floorplan data with 'spaces' key
            save_path (str): Path to save the visualization
            dpi (int): DPI for saved image. If None, calculated for fixed resolution.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Calculate DPI to ensure fixed resolution output
            if dpi is None:
                # Calculate DPI needed for exact resolution
                # resolution = figsize_inches * dpi
                figsize_inches = self.figsize[0]  # assuming square
                dpi = self.resolution / figsize_inches
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Draw rooms
            spaces = floorplan_data.get('spaces', [])
            for space in spaces:
                polygon = space.get('floor_polygon', [])
                if polygon:
                    x_coords = [p['x'] for p in polygon]
                    y_coords = [p['y'] for p in polygon]
                    room_type = space.get('room_type', 'unknown')
                    color = self.room_colors.get(room_type, self.room_colors['unknown'])
                    ax.fill(x_coords, y_coords, color=color, edgecolor=self.border_color, 
                           linewidth=self.linewidth)
            

            
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
            
            # Save with fixed resolution - no tight_layout to ensure consistent size
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0, 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"Warning: Could not generate direct visualization: {e}")
            plt.close()
            return False
