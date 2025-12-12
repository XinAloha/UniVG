import math
import random
from  PriorityQueue import  PriorityQueue
from Attractor import Attractor  # Assuming Attractor is a class defined elsewhere
from Node import Node  # Assuming Node is a class defined elsewhere
from shapely.geometry import Point, Polygon
import numpy as np

class Network:
    def __init__(self, ctx, settings, modality):
        self.ctx = ctx
        self.settings = settings

        self.attractors = []
        self.nodes = []

        self.nodesIndex = None
        self.bounds = []
        self.obstacles = []
        self.pq = PriorityQueue()
        self.modality = modality
        self.IterNum = 0

        # self.buildSpatialIndices()

    def update(self):
        #
        # import time
        # start = time.time()

        for attractor in self.attractors:
            # Associate attractors with nearby nodes
            if self.settings['VenationType'] == 'Open':
                closest_node = self.get_closest_node(attractor, self.get_nodes_in_attraction_zone(attractor))
                if closest_node:
                    closest_node.influencedBy.append(attractor) #
                    attractor.influencingNodes.append(closest_node)
            elif self.settings['VenationType'] == 'Closed':
                neighborhood_nodes = self.get_relative_neighbor_nodes(attractor)
                nodes_in_kill_zone = self.get_nodes_in_kill_zone(attractor)
                nodes_to_grow = [node for node in neighborhood_nodes if node not in nodes_in_kill_zone]

                attractor.influencingNodes += neighborhood_nodes

                if nodes_to_grow:
                    attractor.fresh = False
                    for node in nodes_to_grow:
                        node.influencedBy.append(attractor)
        #print('attractors over')

        # for node in self.nodes:
        #     if node.influencedBy:
        #         average_direction = self.get_average_direction(node, node.influencedBy)
        #         next_node = node.getNextNode(average_direction)
        #         self.add_node(next_node)
        #         node.influencedBy_size = len(node.influencedBy)
        #         node.influencedBy = []
        #     # Canalize nodes: traverse each tip node, check if parent thickness is less than current node thickness plus a fixed value, if so, increase parent thickness, continue to root
        #     if node.isTip and self.settings['EnableCanalization']:
        #         current_node = node
        #         while current_node.parent:
        #             if current_node.parent.thickness < current_node.thickness + 1 and current_node.parent.thickness <= 9:
        #                 current_node.parent.thickness = current_node.thickness + 0.1
        #             current_node = current_node.parent

        for node in self.nodes:
            if node.influencedBy:
                average_direction = self.get_average_direction(node, node.influencedBy)
                next_node = node.getNextNode(average_direction)
                self.add_node(next_node)
                node.influencedBy_size = len(node.influencedBy)
                node.influencedBy = []
                # Canalize nodes: traverse each tip node, check if parent thickness is less than current node thickness plus a fixed value, if so, increase parent thickness, continue to root
            if node.isTip and self.settings['EnableCanalization'] and self.modality == "BrainDsa":
                current_node = node
                while current_node.parent:
                    if current_node.parent.thickness < current_node.thickness + 1 and current_node.parent.thickness <= 12:
                        current_node.parent.thickness = current_node.thickness + 0.1
                    current_node = current_node.parent

        if self.modality == "CoronaryArtery":            
            for node in self.nodes:
                if node.influencedBy_size > 200:
                    node.thickness = node.influencedBy_size * 0.005 + len(node.killAttractors) * 0.01 + 1
                else:
                    node.thickness = node.influencedBy_size * 0.02 + len(node.killAttractors) * 0.01 + 1
        else:
            
            for node in self.nodes:
                node.thickness = 2
        #print('thickness over')

        for attractor in self.attractors:
            if self.settings['VenationType'] == 'Open':
                if attractor.reached:
                    self.attractors.remove(attractor)

            elif self.settings['VenationType'] == 'Closed':
                if not attractor.fresh:
                    all_nodes_reached = all(
                        math.sqrt((node.position[0] - attractor.position[0]) ** 2 + (node.position[1] - attractor.position[1]) ** 2)
                        <= self.settings['KillDistance'] for node in
                        attractor.influencingNodes)
                    if all_nodes_reached:
                        self.attractors.remove(attractor)
        #print('kill attractor over')

    def update_1(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)

        points, = ax.plot([], [], 'bo', markersize=3)

        self.pq.push(self.nodes[0], 0) # add root
        last = 1
        count = 1
        while not self.pq.is_empty() or last < len(self.nodes):
            if count % 100 == 0:
                p = []
                for node in self.nodes:
                    p.append(node.position)
                myset = set(p)
                print(f'list have {len(myset)} different points')
            count += 1
            if count > 5000:
                break
            print(f'priority queue length:{self.pq.length()}')
            last = len(self.nodes)
            # priority reset
            # for node in self.nodes:
            #     node.influencedBy.clear()
            for attractor in self.attractors:
                # Associate attractors with nearby nodes
                if self.settings['VenationType'] == 'Open':
                    closest_node = self.get_closest_node(attractor, self.get_nodes_in_attraction_zone(attractor))
                    if closest_node:
                        closest_node.influencedBy.append(attractor) #
                        attractor.influencingNodes.append(closest_node)
                        if self.pq.contains(closest_node):
                            self.pq.remove(closest_node)
                        self.pq.push(closest_node, len(closest_node.influencedBy))
                elif self.settings['VenationType'] == 'Closed':
                    neighborhood_nodes = self.get_relative_neighbor_nodes(attractor)
                    nodes_in_kill_zone = self.get_nodes_in_kill_zone(attractor)
                    nodes_to_grow = [node for node in neighborhood_nodes if node not in nodes_in_kill_zone]

                    attractor.influencingNodes += neighborhood_nodes

                    if nodes_to_grow:
                        attractor.fresh = False
                        for node in nodes_to_grow:
                            node.influencedBy.append(attractor)
            node = self.pq.peek()
            used = True
            for attractor in node.influencedBy:
                if not attractor.reached:
                    used = False
                    break
            if used:
                self.pq.pop()
            if node.influencedBy:
                average_direction = self.get_average_direction(node, node.influencedBy)
                next_node = node.getNextNode(average_direction)
                self.add_node(next_node)
                node.influencedBy_size += len(node.influencedBy)
                node.influencedBy = []
            print(f'add node len:{len(self.nodes)}')
            # Canalize nodes: traverse each tip node, check if parent thickness is less than current node thickness plus a fixed value, if so, increase parent thickness, continue to root
            for attractor in self.attractors:
                if self.settings['VenationType'] == 'Open':
                    if attractor.reached:
                        self.attractors.remove(attractor)

                elif self.settings['VenationType'] == 'Closed':
                    if not attractor.fresh:
                        all_nodes_reached = all(
                            math.sqrt((node.position[0] - attractor.position[0]) ** 2 + (node.position[1] - attractor.position[1]) ** 2)
                            <= self.settings['KillDistance'] for node in
                            attractor.influencingNodes)
                        if all_nodes_reached:
                            self.attractors.remove(attractor)
            x, y = [], []
            for node in self.nodes:
                x.append(node.position[1])
                y.append(512 - node.position[0])
            points.set_data(x, y)
            plt.pause(0.5)
            plt.draw()

        print('start canalization')
        for node in self.nodes:
            if node.influencedBy_size > 200:
                node.thickness = node.influencedBy_size * 0.005 + len(node.killAttractors) * 0.01 + 1
            else:
                node.thickness = node.influencedBy_size * 0.02 + len(node.killAttractors) * 0.01 + 1
            print(node.influencedBy_size, len(node.killAttractors), node.thickness)
            # if node.isTip and self.settings['EnableCanalization']:
            #     current_node = node
            #     while current_node.parent:
            #         if current_node.parent.thickness <= 9:
            #             current_node.parent.thickness = current_node.thickness + 0.1
            #         current_node = current_node.parent

    def get_relative_neighbor_nodes(self, attractor):
        nearby_nodes = self.get_nodes_in_attraction_zone(attractor)
        relative_neighbors = []

        for p0 in nearby_nodes:
            fail = False
            #attractor_to_p0 = p0.position - attractor.position
            attractor_to_p0 = self.tuple_subtraction(p0.position, attractor.position)
            for p1 in nearby_nodes:
                if p0 == p1:
                    continue

                #attractor_to_p1 = p1.position - attractor.position
                attractor_to_p1 = self.tuple_subtraction(p1.position, attractor.position)
                attractor_to_p1_length = math.sqrt(attractor_to_p1[0] ** 2 + attractor_to_p1[1] ** 2)
                attractor_to_p0_length = math.sqrt(attractor_to_p0[0] ** 2 + attractor_to_p0[1] ** 2)
                if attractor_to_p1_length > attractor_to_p0_length:
                    continue

                #p0_to_p1 = p1.position - p0.position
                p0_to_p1 = self.tuple_subtraction(p1.position, p0.position)
                p0_to_p1_length = math.sqrt(p0_to_p1[0] ** 2 + p0_to_p1[1] ** 2)
                if attractor_to_p0_length > p0_to_p1_length:
                    fail = True
                    break

            if not fail:
                relative_neighbors.append(p0)

        return relative_neighbors

    # Get all nodes within attractor's attraction zone
    def get_nodes_in_attraction_zone(self, attractor):
        Nodes = []
        for node in self.nodes:
                if math.sqrt((attractor.position[0] - node.position[0]) ** 2 + (attractor.position[1] - node.position[1]) ** 2) <= self.settings['AttractionDistance']:
                    Nodes.append(node)
        return Nodes

    # Get nodes within kill zone
    def get_nodes_in_kill_zone(self, attractor):
        Nodes = []
        for node in self.nodes:
            if math.sqrt((attractor.position[0] - node.position[0]) ** 2 + (attractor.position[1] - node.position[1]) ** 2) <= self.settings['KillDistance']:
                Nodes.append(node)
        return Nodes

    # Get closest node to attractor
    def get_closest_node(self, attractor, nearby_nodes):
        closest_node = None
        record = self.settings['AttractionDistance']

        for node in nearby_nodes:
            distance = math.sqrt((attractor.position[0] - node.position[0]) ** 2 + (attractor.position[1] - node.position[1]) ** 2)

            # Ignore attractors within node's kill distance
            if distance < self.settings['KillDistance']:
                attractor.reached = True
                node.killAttractors.append(attractor)
                closest_node = None
            elif distance < record:
                closest_node = node
                record = distance

        return closest_node

    def tuple_subtraction(self, tuple1, tuple2):
        return tuple(x - y for x, y in zip(tuple1, tuple2))

    def tuple_division(self,tuple1, divisor):
        return tuple(x / divisor for x in tuple1)

    def tuple_sum(self, tuple1, tuple2):
        return tuple(x + y for x, y in zip(tuple1, tuple2))

    # Get average direction
    def get_average_direction(self, node, nearby_attractors):
        average_direction = [0, 0]
        from numpy.linalg import norm
        for attractor in nearby_attractors:
            #direction_to_attractor = attractor.position - node.position
            direction_to_attractor = self.tuple_subtraction(attractor.position, node.position)
            #normalized_direction = direction_to_attractor / norm(direction_to_attractor)
            normalized_direction = self.tuple_division(direction_to_attractor, norm(direction_to_attractor))
            #average_direction += normalized_direction
            average_direction = self.tuple_sum(average_direction, normalized_direction)
        #average_direction += random.randint(-1, 2, size=(2,))
        # Remove randomness
        average_direction = self.tuple_sum(average_direction, tuple(np.random.randint(-1, 2, size=(2,))))
        #average_direction /= len(nearby_attractors)
        average_direction = self.tuple_division(average_direction, len(nearby_attractors))

        return average_direction

    # Add node
    def add_node(self, node):
        is_inside_any_bounds = False
        is_inside_any_obstacle = False

        if self.bounds and len(self.bounds) > 0:
            polygon = Polygon(self.bounds)
            point = Point(node.position[0], node.position[1])
            if polygon.touches(point):
                is_inside_any_bounds = True

        if self.obstacles and len(self.obstacles) > 0:
            polygon = Polygon(self.obstacles)
            point = Point(node[0], node[1])
            if  polygon.touches(point):
                is_inside_any_obstacle = True

        if (is_inside_any_bounds or not self.bounds) and \
           (not is_inside_any_obstacle or not self.obstacles):
            self.nodes.append(node)

    def reset(self):
        self.nodes = []
        self.attractors = []

