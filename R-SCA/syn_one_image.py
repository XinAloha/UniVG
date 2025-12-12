import os

from Network import Network
from Attractor import Attractor
from Node import Node
from AttractorPatterns import  get_random_attractors, get_grid_of_attractors, get_artery_attractors, get_nerveFiber_attractors
import yaml
import cv2
import numpy as np
from PIL import Image
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import turtle
from PIL import Image
import random
from tqdm import tqdm

def find_start_point(image):
    upbound = []
    x, y = image.shape
    s, e = -1, -1
    max_len = -1
    ans = None
    max_ans = -1
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
        max_ans = max(max_ans, ans1[1])
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
        max_ans = max(max_ans, ans2[0])
        f2 = True

    s, e = -1, -1
    max_len = -1
    ans = None
    for i in range(x):
        if image[i][y - 1] != 0 and s == -1:
            s = i
            e = i
            while e < x and image[e][y - 1] != 0:
                e += 1
            e -= 1
            if max_len < (e - s + 1):
                max_len = e - s + 1
                ans = (s, e)
        i = e + 1
    f3 = False
    if  ans != None:
        ans3 = (int((ans[0] + ans[1]) / 2.0), y - 1)
        max_ans = max(max_ans, ans3[0])
        f3 = True

    s, e = -1, -1
    max_len = -1
    ans = None
    for i in range(y):
        if image[x - 1][i] != 0 and s == -1:
            s = i
            e = i
            while e < y and image[x - 1][e] != 0:
                e += 1
            e -= 1
            if max_len < (e - s + 1):
                max_len = e - s + 1
                ans = (s, e)
        i = e + 1
    f4 = False
    if ans != None:
        ans4 = (x - 1, int((ans[0] + ans[1]) / 2.0))
        max_ans = max(max_ans, ans4[1])
        f4 = True
    if f1 and max_ans == ans1[1]:
        return ans1
    elif f2 and max_ans == ans2[0]:
        return ans2
    elif f3 and max_ans == ans3[0]:
        return ans3
    elif f4 and max_ans == ans4[1]:
        return ans4
    print('position err')
    return (0, 0)

def change_worldcoordinates(tuple, weight=512, height=512):
    return (height - height + tuple[1], -tuple[0] + weight)

def draw_tree(node,parent=None, amount=0):

    if parent:
        thickness = node.thickness
        plt.plot([parent.position[1], node.position[1]], [512-parent.position[0], 512-node.position[0]], 'w', lw=thickness)
    for child in node.children:
       draw_tree(child, node,amount=amount)
    

def draw(roots, count, modality):
    plt.figure(figsize=(512/100  , 512/100), dpi=100)
    plt.gca().set_facecolor('black')  # 设置背景为黑色

    plt.axis('off')
    if modality=="NerveFiber":
        for r in root:
   
            draw_tree(r, parent=None, amount=0)
    else:
        draw_tree(root, parent=None, amount=0)
    plt.savefig('E:\Project\SpaceClone\synthesisLabel\BrainDSA\\' + str(count) + '.png', bbox_inches='tight', dpi=100, pad_inches=0, facecolor='black')  # 设置保存图像时的背景颜色
    plt.clf() #清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。
    plt.close() #完全关闭图形窗口

def getArteriaCoronariaBounds(path):

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image_arr = np.array(image)
    sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    x, y = image_arr.shape
    root = find_start_point(image_arr)
    gradient_magnitude = ((gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude))) * 255
    gradient_magnitude = np.where(gradient_magnitude > 125, 1, 0)
    image = Image.fromarray(gradient_magnitude.astype('uint8'))
    bound = []
    for i in range(x):
        for j in range(y):
            if image_arr[i][j] != 0:
                bound.append((i, j))
    return bound, root,  gradient_magnitude

def getNerveFiberRoots(path):
        # 读取图像  
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
      
    # 二值化图像（这里假设我们已经有了一个二值图像，或者通过阈值化获得）  
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)  
      
    # 执行闭运算：先膨胀后腐蚀  
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  
      
    # 查找轮廓  
    contours, _ = cv2.findContours(closing , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
    # 存储每个连通区域的最靠上面的点  
    top_points = []  
      
    # 遍历每个轮廓  
    for contour in contours:  
        # 遍历轮廓上的每个点，找到y坐标最小的点  
        min_y = np.inf  
        min_point = None  
        for point in contour:  
            x, y = point.ravel()  
            if y < min_y:  
                min_y = y  
                min_point = (y, x)  

        # 将点添加到列表中  
        # print("min_point:", )
        top_points.append(min_point)  
       

    # for point in top_points:  
    #     cv2.circle(closing, point, 10, (255, 125, 0), -1)  
    # cv2.imshow('Random Points', closing)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()  
      
    return top_points
def resetNetwork(network, ctx, settings, path, modality):
    network.reset()
    if modality == "CoronaryArtery" or modality == "Brain":
        network.bounds, root, _ = getArteriaCoronariaBounds(path)
        polygon = Polygon(network.bounds)
        point = Point(root[0], root[1])

        root = Node(None, root, isTip=True, ctx=ctx, settings=settings)
        network.add_node(root)
        randomAttractors = get_random_attractors(num_attractors=100, ctx=ctx, settings=settings,bounds=network.bounds, obstacles=None)
        gridAttractors = get_grid_of_attractors(
                                                num_rows=50, 
                                                num_columns=50,
                                                ctx= ctx, 
                                                settings=settings, 
                                                jitter_range=0, 
                                                bounds=network.bounds, 
                                                obstacles=None
                                                )
        arteryAttractors = get_artery_attractors(network, ctx)
        random_num = random.randint(1, 3)
        # random_num = 2
        network.attractors = arteryAttractors
        if random_num == 1:
            network.attractors += gridAttractors
        elif random_num == 2:
            network.attractors += randomAttractors
        elif random_num == 3:
            network.attractors = network.attractors + gridAttractors + randomAttractors
        return root
    elif modality == "NerveFiber":
        roots = getNerveFiberRoots(path)
        temproots = []
        for root in roots:
             root = Node(None, root, isTip=True, ctx=ctx, settings=settings)
             temproots.append(root)
             network.add_node(root)
        roots = temproots
        img = Image.open(path).convert("L")
        img = np.array(img)
        imgAttractors = get_nerveFiber_attractors(network, ctx, img, roots)


        random_num = random.randint(1, 2)
        #random_num = 1
        #print("random_num:", random_num)
        network.attractors = imgAttractors
        if random_num == 1:
            num_rows = random.randint(10, 20)
            num_columns = num_rows
            gridAttractors = get_grid_of_attractors(
                                                num_rows=num_rows, 
                                                num_columns=num_columns,
                                                ctx= ctx, 
                                                settings=settings, 
                                                jitter_range=0, 
                                                bounds=network.bounds, 
                                                obstacles=None
                                                )
            network.attractors += gridAttractors
        elif random_num == 2:
            randomAttractors = get_random_attractors(num_attractors=random.randint(50, 100), ctx=ctx, settings=settings,bounds=network.bounds, obstacles=None)
            network.attractors += randomAttractors
        elif random_num == 3:
            randomAttractors = get_random_attractors(num_attractors=random.randint(50, 100), ctx=ctx, settings=settings,bounds=network.bounds, obstacles=None)
            num_rows = random.randint(50, 150)
            num_columns = num_rows
            gridAttractors = get_grid_of_attractors(
                                                num_rows=num_rows, 
                                                num_columns=num_columns,
                                                ctx= ctx, 
                                                settings=settings, 
                                                jitter_range=0, 
                                                bounds=network.bounds, 
                                                obstacles=None
                                                )
            network.attractors = network.attractors + gridAttractors + randomAttractors
        return roots
    return None

if __name__ == "__main__":
    count = 1
    # load config
    with open('paper.yaml', 'r') as file:
        settings = yaml.safe_load(file)

    canvas = np.zeros((512, 512))
    path = r"E:\Project\SpaceClone\label\Brain"
    images = os.listdir(path)
    while count != 2000:
        for image in images:
            image_path = os.path.join(path, image)
            modality = "Brain"
            network = Network(canvas, settings, modality=modality)
            root = resetNetwork(network, canvas, settings, image_path, modality)
            # break
        
            BFS = True
            Display = True
            if BFS:
                last = 0
                if Display:
                    fig, ax = plt.subplots()
                    ax.set_xlim(0, 512)
                    ax.set_ylim(0, 512)
                    points, = ax.plot([], [], 'bo', markersize=2)
                iterationAmount = 0
                while True:
                    network.update()
                    iterationAmount += 1
                    #print("iteration:", iterationAmount, last, len(network.nodes))
                    if len(network.nodes) == last:
                        print("迭代次数：", iterationAmount)
                        print('space colone over')
                        break
                    
                    last = len(network.nodes)
                    if Display:
                       
                        x, y = [], []
                        for node in network.nodes:
                            x.append(node.position[1])
                            y.append(512-node.position[0])
    
                        points.set_data(x, y)
                        plt.pause(0.2)
                        plt.draw()
                    # if iterationAmount >= 2:
                    #     break
                    #plt.show()
            
                draw(root, count, modality=modality)

                count += 1

                # 随着迭代次数的增加, 
                # 每个节点寻找下一个子节点的长度（SegmentLength）应该逐渐变小，一直到最后一个最小值能保证拟合曲线
                # 同时 吸引子范围也要缩小，保证下一节点尽可能在周围被找到， 前两者缩小，节点杀伤范围也要缩小
                # 脑部血管：严格满足树的概念，主干粗，枝干细
                # 初始 SegmentLength:  15          
                #      AttractionDistance: 20       
                #      KillDistance: 10  
