import random
from Attractor import Attractor
from shapely.geometry import Point, Polygon

def get_random_attractors(num_attractors, ctx, settings, bounds=None, obstacles=None):
    attractors = []
    is_inside_any_bounds = False
    is_inside_any_obstacle = False
    is_on_screen = False
    polygon = Polygon(bounds)
    for i in range(num_attractors):
        x = random.randint(0, settings['Width'])
        y = random.randint(0, settings['Height'])
        is_inside_any_bounds = False
        is_inside_any_obstacle = False
        is_on_screen = False

        # Only allow attractors that are in the viewport
        if 0 < x < settings['Width'] and 0 < y < settings['Height']:
            is_on_screen = True

        point = Point(x, y)
        # Only allow attractors inside defined bounds
        if bounds and len(bounds) > 0:
            if polygon.touches(point):
                is_inside_any_bounds = True

        # Don't allow attractors inside obstacles
        if obstacles and len(obstacles) > 0:
            for obstacle in obstacles:
                if obstacle.touches(x, y):
                    is_inside_any_obstacle = True

        if settings["Modality"] == "OCT" or ((is_inside_any_bounds or not bounds) and \
           (not is_inside_any_obstacle or not obstacles) and \
           is_on_screen):
            attractors.append(
                Attractor(
                    (x, y),
                    ctx
                )
            )

    return attractors

def get_artery_attractors(network, ctx):
    attractors = []
    for point in network.bounds:
        attractors.append(Attractor(
            (point[0], point[1]),
            ctx
        ))
    return attractors

def get_OCT_attractors(network, ctx,img, roots):
    attractors = []
    x, y = img.shape
    for i in range(x):
        for j in range(y):
            if img[i][j] != 0:
                attractors.append(
                    Attractor(
                        (i, j),
                        ctx
                    )
                )
    for root in roots:
        attractors.append(
                    Attractor(
                        root.position,
                        ctx
                    )
                )
    return attractors


def get_grid_of_attractors(num_rows, num_columns, ctx, settings, jitter_range=10, bounds=None, obstacles=None):
    attractors = []
    is_inside_any_bounds = False
    is_inside_any_obstacle = False
    is_on_screen = False
    from tqdm import tqdm
    polygon = Polygon(bounds)
    for i in range(num_rows + 1):
        for j in range(num_columns + 1):
            x = int(settings['Width'] / num_rows * i) + random.randint(-jitter_range, jitter_range)
            y = int(settings['Height'] / num_columns * j) + random.randint(-jitter_range, jitter_range)
            is_inside_any_bounds = False
            is_inside_any_obstacle = False
            is_on_screen = False

            # Only allow attractors that are in the viewport
            if 0 < x < settings['Width'] and 0 < y < settings['Height']:
                is_on_screen = True

            # Only allow attractors within bounds region
            if bounds and len(bounds) > 0:
                point = Point(x, y)
                if polygon.touches(point):
                        is_inside_any_bounds = True

            # Don't allow attractors inside obstacles
            if obstacles and len(obstacles) > 0:
                for obstacle in obstacles:
                    if obstacle.touches(x, y):
                        is_inside_any_obstacle = True

            if settings["Modality"] == "OCT" or (is_on_screen and \
               (is_inside_any_bounds or not bounds) and \
               (not is_inside_any_obstacle or not obstacles)):
                attractors.append(
                    Attractor(
                        (x, y),
                        ctx
                    )
                )

    return attractors


