import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue
import time

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
    

def draw(roots, count, modality, output_dir='E:\Project\SpaceClone\SynthesisLabel\SBCD1'):
    plt.figure(figsize=(512/100  , 512/100), dpi=100)
    plt.gca().set_facecolor('black')  # 设置背景为黑色

    plt.axis('off')
    if modality=="NerveFiber":
        for r in roots:
            draw_tree(r, parent=None, amount=0)
    else:
        draw_tree(roots, parent=None, amount=0)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{count}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=100, pad_inches=0, facecolor='black')
    plt.clf() #清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。
    plt.close() #完全关闭图形窗口
    return output_path

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
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
      
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

def process_single_image(args):
    """处理单张图像的函数，用于并发执行"""
    image_path, modality, settings, count_value, output_dir = args
    
    try:
        # 设置matplotlib为非交互模式，避免并发冲突
        plt.ioff()
        
        canvas = np.zeros((512, 512))
        network = Network(canvas, settings, modality=modality)
        root = resetNetwork(network, canvas, settings, image_path, modality)
        
        if root is None:
            return None, f"Failed to initialize network for {image_path}"
        
        # 执行网络生成
        iterationAmount = 0
        last = 0
        
        while True:
            network.update()
            iterationAmount += 1
            
            if len(network.nodes) == last:
                break
            last = len(network.nodes)
            
            # 防止无限循环
            if iterationAmount > 10000:
                break
        
        # 绘制并保存结果
        output_path = draw(root, count_value, modality, output_dir)
        
        return output_path, f"Successfully processed {os.path.basename(image_path)} -> {os.path.basename(output_path)}"
        
    except Exception as e:
        return None, f"Error processing {image_path}: {str(e)}"

class CounterManager:
    """线程安全的计数器管理器"""
    def __init__(self, start_value=1):
        self._value = start_value
        self._lock = threading.Lock()
    
    def get_next(self):
        with self._lock:
            current = self._value
            self._value += 1
            return current
    
    def get_current(self):
        with self._lock:
            return self._value

def process_images_concurrent(images_dir, modality, settings, max_images=1000, 
                            max_workers=None, output_dir='E:\Project\SpaceClone\SynthesisLabel\SBCD3'):
    """并发处理图像的主函数"""
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)  # 限制最大进程数，避免内存溢出
    
    print(f"使用 {max_workers} 个进程进行并发处理")
    
    # 获取所有图像文件
    images = os.listdir(images_dir)
    image_paths = [os.path.join(images_dir, img) for img in images]
    
    # 创建计数器管理器
    counter = CounterManager(1)
    
    # 准备任务参数
    tasks = []
    total_tasks = 0
    
    # 重复处理直到达到最大图像数
    while total_tasks < max_images:
        for image_path in image_paths:
            if total_tasks >= max_images:
                break
            
            count_value = counter.get_next()
            task_args = (image_path, modality, settings, count_value, output_dir)
            tasks.append(task_args)
            total_tasks += 1
    
    print(f"准备处理 {len(tasks)} 个任务")
    
    # 使用进度条和进程池执行任务
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=len(tasks), desc="处理图像") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_path, message = future.result()
                    if result_path:
                        successful += 1
                        pbar.set_postfix({"成功": successful, "失败": failed})
                    else:
                        failed += 1
                        print(f"任务失败: {message}")
                        pbar.set_postfix({"成功": successful, "失败": failed})
                except Exception as e:
                    failed += 1
                    print(f"任务执行异常: {str(e)}")
                    pbar.set_postfix({"成功": successful, "失败": failed})
                
                pbar.update(1)
    
    print(f"\n处理完成! 成功: {successful}, 失败: {failed}")
    return successful, failed

def main():
    """主函数，用于支持多进程"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='Spatial Colonization Algorithm (SCA) for vascular structure synthesis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing real vascular masks')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for generated synthetic images')
    parser.add_argument('--modality', type=str, default='CoronaryArtery',
                        choices=['CoronaryArtery', 'Brain', 'NerveFiber'],
                        help='Vascular modality type')
    parser.add_argument('--config', type=str, default='./paper.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--max-images', type=int, default=2000,
                        help='Maximum number of images to generate')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of worker processes (None for auto)')
    parser.add_argument('--concurrent', action='store_true', default=True,
                        help='Use concurrent processing mode')
    parser.add_argument('--serial', dest='concurrent', action='store_false',
                        help='Use serial processing mode (disables concurrent)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as file:
        settings = yaml.safe_load(file)
    
    # 设置参数
    images_dir = args.input_dir
    output_dir = args.output_dir
    modality = args.modality
    MAX_IMAGES = args.max_images
    MAX_WORKERS = args.max_workers
    USE_CONCURRENT = args.concurrent
    
    print(f"开始处理图像，模态: {modality}")
    print(f"输入目录: {images_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标生成数量: {MAX_IMAGES}")
    print(f"处理模式: {'并发' if USE_CONCURRENT else '串行'}")
    if MAX_WORKERS:
        print(f"工作进程数: {MAX_WORKERS}")
    
    start_time = time.time()

    if USE_CONCURRENT:
        print("=== 使用并发模式 ===")
        successful, failed = process_images_concurrent(
            images_dir=images_dir,
            modality=modality,
            settings=settings,
            max_images=MAX_IMAGES,
            max_workers=MAX_WORKERS,
            output_dir=output_dir
        )
        print(f"并发处理结果: 成功 {successful}, 失败 {failed}")
        
    else:
        print("=== 使用原始串行模式 ===")
        count = 1
        canvas = np.zeros((512, 512))
        images = os.listdir(images_dir)
        
        with tqdm(total=MAX_IMAGES, desc="串行处理图像") as pbar:
            while count <= MAX_IMAGES:
                for image in images:
                    if count > MAX_IMAGES:
                        break
                        
                    image_path = os.path.join(images_dir, image)
                    network = Network(canvas, settings, modality=modality)
                    root = resetNetwork(network, canvas, settings, image_path, modality)
                    
                    if root is None:
                        continue
                    
                    # 执行网络生成
                    iterationAmount = 0
                    last = 0
                    
                    while True:
                        network.update()
                        iterationAmount += 1
                        
                        if len(network.nodes) == last:
                            break
                        last = len(network.nodes)
                        
                        # 防止无限循环
                        if iterationAmount > 10000:
                            break
                    
                    # 绘制并保存
                    draw(root, count, modality=modality, output_dir=output_dir)
                    count += 1
                    pbar.update(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n总处理时间: {elapsed_time:.2f} 秒")
    print(f"平均每张图像处理时间: {elapsed_time/MAX_IMAGES:.2f} 秒")
    
    if USE_CONCURRENT:
        print(f"并发模式使用了 {MAX_WORKERS or min(mp.cpu_count(), 4)} 个进程")
    
    print("处理完成!")

if __name__ == "__main__":
    # 对于Windows系统的多进程支持
    mp.freeze_support()
    main()