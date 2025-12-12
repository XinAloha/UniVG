import os

from Network import Network
from Attractor import Attractor
from Node import Node
from AttractorPatterns import  get_random_attractors, get_grid_of_attractors, get_artery_attractors
import yaml
import cv2
import numpy as np
from PIL import Image
#from skimage import measure
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import turtle
from PIL import Image

count = 0
def find_start_point(image):
    upbound = []
    x, y = image.shape
    s, e = -1, -1
    max_len = -1
    ans = None
    for i in range(y):
        if image[0][i] != 0 and s == -1:
            s = i
            e = i
            while e < y and image[0][e] != 0:
                e += 1
            e -= 1
            if max_len < (e - s + 1):
                max_len = e - s + 1
                ans = (s, e)
        i = e + 1
    f1 = False
    if ans != None:
        ans1 = (0, int((ans[0] + ans[1]) / 2.0))
        f1 = True
    s, e = -1, -1
    max_len = -1
    ans = None
    for i in range(x):
        if image[i][0] != 0 and s == -1:
            s = i
            e = i
            while e < x and image[e][0] != 0:
                e += 1
            e -= 1
            if max_len < (e - s + 1):
                max_len = e - s + 1
                ans = (s, e)
        i = e + 1
    f2 = False
    if  ans != None:
        ans2 = (int((ans[0] + ans[1]) / 2.0), 0)
        f2 = True

    if f1 and f2:
        return ans1 if ans1[1] > ans2[0] else ans2
    if not f1 and f2:
        return ans2
    if not f2 and f1:
        return ans1
    print('position err')
    return (0, 0)

def change_worldcoordinates(tuple, weight=512, height=512):
    return (height - height + tuple[1], -tuple[0] + weight)

def draw_tree(node,parent=None,weight=512, height=512):
    if parent:
        thickness = node.thickness
        plt.plot([parent.position[1], node.position[1]], [512-parent.position[0], 512-node.position[0]], 'w', lw=thickness)
        #print(f'node thickness: {node.thickness}, node kill attractors: {len(node.killAttractors)}, node affect attractors:{node.influencedBy_size}')
    for child in node.children:
        draw_tree(child, node)

def draw(root, count):
    plt.figure(figsize=(512 / 72, 512 / 72))
    plt.gca().set_facecolor('black')  # 设置背景为黑色

    plt.axis('off')
    draw_tree(root)

    plt.savefig('../synthesis_label/'+str(count)+'.png', bbox_inches='tight', dpi=512, facecolor='black')  # 设置保存图像时的背景颜色
def getArteriaCoronariaBounds(path, display=False):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_arr = np.array(image)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    x, y = image_arr.shape
    root = find_start_point(image_arr)

    #gradient_magnitude = np.flipud(gradient_magnitude)
    gradient_magnitude = ((gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude))) * 255
    gradient_magnitude = np.where(gradient_magnitude > 125, 1, 0)
    image = Image.fromarray(gradient_magnitude.astype('uint8'))
    bound = []
    for i in range(x):
        for j in range(y):
            if image_arr[i][j] != 0:
                bound.append((i, j))
    return bound, root,  gradient_magnitude

def resetNetwork(network, ctx, settings, path):
    network.reset()
    network.bounds, root, _ = getArteriaCoronariaBounds(path)
    polygon = Polygon(network.bounds)
    point = Point(root[0], root[1])

    print('root position', root)
    root = Node(None, root, isTip=True, ctx=ctx, settings=settings)
    network.add_node(root)
    #randomAttractors = get_random_attractors(num_attractors=500, ctx=ctx, settings=settings,bounds=network.bounds, obstacles=None)
    #gridAttractors = get_grid_of_attractors(num_rows=300, num_columns=300,ctx= ctx, settings=settings, jitter_range=0, bounds=network.bounds, obstacles=None)
    arteryAttractors = get_artery_attractors(network, ctx)
    network.attractors = arteryAttractors
    return root



if __name__ == "__main__":
    path = '../label'
    image_path_list = os.listdir(path)[195:]
    count = 195
    from tqdm import tqdm
    print(image_path_list[5])
    # for image_name in tqdm(image_path_list):
    #     count += 1
    #     canvas = np.zeros((512, 512))
    #     image_path = os.path.join(path, image_name)
    #     print(image_path)
    #     with open('./Defaults.yaml', 'r') as file:
    #         settings = yaml.safe_load(file)
    #
    #     network = Network(canvas, settings)
    #     root = resetNetwork(network, canvas, settings, image_path)
    #
    #     BFS = True
    #     if BFS:
    #         last = 0
    #         # fig, ax = plt.subplots()
    #         # ax.set_xlim(0, 512)
    #         # ax.set_ylim(0, 512)
    #         # points, = ax.plot([], [], 'bo', markersize=2)
    #         while True:
    #             network.update()
    #             # print('update', len(network.nodes))
    #             if len(network.nodes) == last:
    #                 print('space colone over')
    #                 break
    #
    #             last = len(network.nodes)
    #         #     x, y = [], []
    #         #     for node in network.nodes:
    #         #         x.append(node.position[1])
    #         #         y.append(512-node.position[0])
    #         #     points.set_data(x, y)
    #         #     plt.pause(0.5)
    #         #     plt.draw()
    #         # plt.show()
    #     else:
    #         network.update_1() # DFS
    #     draw(root, count)
    #
    #

