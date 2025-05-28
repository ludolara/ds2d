from matplotlib.patches import Rectangle, Arc
from shapely.geometry import Polygon
import re


def plot_floorplan(floorplan, font_size=8, with_size=True):
    rooms, doors, windows = floorplan["rooms"], floorplan["doors"], floorplan["windows"]

    # SETTINGS
    OUTER_WALL_WIDTH = 8

    DOOR_ANGLE = 80
    DOOR_WIDTH = 1


    fig, ax1 = plt.subplots(1, 1, figsize=(15, 7))

    # find all room polygons, load all polygon lines into one big array

    # perimeter
    outer_poly = None

    # replace CamelCase room names with spaced room names
    pattern = re.compile(r'(?<!^)(?=[A-Z])')

    def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in degrees.
        """
        ox, oy = origin
        px, py = point

        angle = math.radians(angle)

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def get_angle(pa, pb):
        myradians = math.atan2(pb[1]-pa[1], pb[0]-pa[0])
        mydegrees = math.degrees(myradians)
        if mydegrees < 0:
            mydegrees = 360 + mydegrees
        return mydegrees

    # Plot floorplan on the left subplot
    for index, room in enumerate(rooms):
        print (room)

        x_coords = [point['x'] for point in room['floor_polygon']]
        y_coords = [point['y'] for point in room['floor_polygon']]

        # Close the polygon by adding the first point at the end
        x_coords.append(x_coords[0])
        y_coords.append(y_coords[0])

        # ax1.plot(x_coords, y_coords, color=colors[index % len(colors)])
        ax1.plot(x_coords, y_coords, color="black", linewidth=2)

        coords = [(pt["x"], pt["y"]) for pt in room["floor_polygon"]]

        # create polygon for room
        poly = Polygon(coords)

        # add polygon to outer poly (for perimeter)
        if outer_poly is None:
            outer_poly = poly
        else:
            outer_poly = outer_poly.union(poly)

        room_name = pattern.sub(' ', room['room_type'])

        if with_size:
            room_name += f"\n{room['area']} m²"

        # print room name
        ax1.text(poly.centroid.x, poly.centroid.y, room_name, fontsize=font_size, horizontalalignment='center', verticalalignment='center')


    # draw outer wall
    x,y = outer_poly.exterior.xy
    ax1.plot(x, y, color="black", linewidth=OUTER_WALL_WIDTH)

    for door in doors:
        start_x = door['position'][0]['x']
        start_y = door['position'][0]['y']
        end_x = door['position'][1]['x']
        end_y = door['position'][1]['y']

        # door radius
        dist = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)

        # door angle
        angle = get_angle((start_x, start_y), (end_x, end_y))

        # rotating the door around the start point to "open" it
        new_end = rotate((start_x, start_y), (end_x, end_y), -DOOR_ANGLE)

        # erasing the wall segment (painting white over it)
        ax1.plot([start_x, end_x], [start_y, end_y], color='white', linewidth=OUTER_WALL_WIDTH, solid_capstyle='butt')

        # drawing the new angled door
        ax1.plot([start_x, new_end[0]], [start_y, new_end[1]], color='grey', linewidth=DOOR_WIDTH)

        # drawing the door opening arc
        arc = Arc((start_x, start_y), dist*2, dist*2, color='grey', theta1=angle-DOOR_ANGLE, theta2=angle, linewidth=DOOR_WIDTH)
        ax1.add_patch(arc)


    for window in windows:
        start_x = window['position'][0]['x']
        start_y = window['position'][0]['y']
        end_x = window['position'][1]['x']
        end_y = window['position'][1]['y']
        ax1.plot([start_x, end_x], [start_y, end_y], color='white', linewidth=OUTER_WALL_WIDTH-2, solid_capstyle='butt' ) # hehe, butt
        ax1.plot([start_x, end_x], [start_y, end_y], color='black', linewidth=1, solid_capstyle='butt' )



    # Setting up labels and title for the floorplan
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.set_aspect('equal', 'box')  # Set aspect ratio to equal
    # ax1.invert_yaxis()  # Invert Y-axis to match ground truth orientation

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()