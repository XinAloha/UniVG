import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import queue
import time

from Network import Network
from Attractor import Attractor
from Node import Node
from AttractorPatterns import  get_random_attractors, get_grid_of_attractors, get_artery_attractors, get_OCT_attractors
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

def draw_tree(node, parent=None, amount=0, image_size=512):

    if parent:
        thickness = node.thickness
        plt.plot([parent.position[1], node.position[1]], [image_size-parent.position[0], image_size-node.position[0]], 'w', lw=thickness)
    for child in node.children:
       draw_tree(child, node, amount=amount, image_size=image_size)
    

def draw(roots, count, modality, output_dir, image_size=512):
    plt.figure(figsize=(image_size/100, image_size/100), dpi=100)
    plt.gca().set_facecolor('black')
    plt.axis('off')
    if modality=="OCT":
        for r in roots:
            draw_tree(r, parent=None, amount=0, image_size=image_size)
    else:
        draw_tree(roots, parent=None, amount=0, image_size=image_size)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{count}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=100, pad_inches=0, facecolor='black')
    plt.clf()
    plt.close()
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

def getOCTRoots(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    kernel_size = (5, 5)
    kernel = np.ones(kernel_size, np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_points = []
    
    for contour in contours:
        min_y = np.inf
        min_point = None
        for point in contour:
            x, y = point.ravel()
            if y < min_y:
                min_y = y
                min_point = (y, x)
        top_points.append(min_point)
    
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
        network.attractors = arteryAttractors
        if random_num == 1:
            network.attractors += gridAttractors
        elif random_num == 2:
            network.attractors += randomAttractors
        elif random_num == 3:
            network.attractors = network.attractors + gridAttractors + randomAttractors
        return root
    elif modality == "OCT":
        roots = getOCTRoots(path)
        temproots = []
        for root in roots:
             root = Node(None, root, isTip=True, ctx=ctx, settings=settings)
             temproots.append(root)
             network.add_node(root)
        roots = temproots
        img = Image.open(path).convert("L")
        img = np.array(img)
        imgAttractors = get_OCT_attractors(network, ctx, img, roots)


        random_num = random.randint(1, 2)
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
    """Process a single image for concurrent execution"""
    image_path, modality, settings, count_value, output_dir, image_size = args
    
    try:
        plt.ioff()
        canvas = np.zeros((image_size, image_size))
        network = Network(canvas, settings, modality=modality)
        root = resetNetwork(network, canvas, settings, image_path, modality)
        
        if root is None:
            return None, f"Failed to initialize network for {image_path}"
        
        iterationAmount = 0
        last = 0
        
        while True:
            network.update()
            iterationAmount += 1
            
            if len(network.nodes) == last:
                break
            last = len(network.nodes)
            
            if iterationAmount > 10000:
                break
        
        output_path = draw(root, count_value, modality, output_dir, image_size)
        return output_path, f"Successfully processed {os.path.basename(image_path)} -> {os.path.basename(output_path)}"
        
    except Exception as e:
        return None, f"Error processing {image_path}: {str(e)}"

class CounterManager:
    """Thread-safe counter manager"""
    def __init__(self, start_value=1):
        self._value = start_value
        self._lock = threading.Lock()
    
    def get_next(self):
        with self._lock:
            current = self._value
            self._value += 1
            return current

def process_images_concurrent(images_dir, modality, settings, max_images=1000, 
                            max_workers=None, output_dir=None, image_size=512):
    """Main function for concurrent image processing"""
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 4)
    
    print(f"Using {max_workers} processes for concurrent processing")
    
    images = os.listdir(images_dir)
    image_paths = [os.path.join(images_dir, img) for img in images]
    counter = CounterManager(1)
    tasks = []
    total_tasks = 0
    
    while total_tasks < max_images:
        for image_path in image_paths:
            if total_tasks >= max_images:
                break
            
            count_value = counter.get_next()
            task_args = (image_path, modality, settings, count_value, output_dir, image_size)
            tasks.append(task_args)
            total_tasks += 1
    
    print(f"Preparing to process {len(tasks)} tasks")
    
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_single_image, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Processing images") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result_path, message = future.result()
                    if result_path:
                        successful += 1
                        pbar.set_postfix({"Success": successful, "Failed": failed})
                    else:
                        failed += 1
                        print(f"Task failed: {message}")
                        pbar.set_postfix({"Success": successful, "Failed": failed})
                except Exception as e:
                    failed += 1
                    print(f"Task exception: {str(e)}")
                    pbar.set_postfix({"Success": successful, "Failed": failed})
                
                pbar.update(1)
    
    print(f"\nProcessing complete! Success: {successful}, Failed: {failed}")
    return successful, failed

def main():
    """Main function with multiprocessing support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Spatial Colonization Algorithm (SCA) for vascular structure synthesis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input-dir', type=str, default=r"./R-SCA/RealCoronaryArteryMask",
                        help='Input directory containing real vascular masks')
    parser.add_argument('--output-dir', type=str, default=r"./R-SCA/output",
                        help='Output directory for generated synthetic images')
    parser.add_argument('--modality', type=str, default='CoronaryArtery',
                        choices=['CoronaryArtery', 'Brain', 'OCT'],
                        help='Vascular modality type')
    parser.add_argument('--config', type=str, default='./R-SCA/paper.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--max-images', type=int, default=2000,
                        help='Maximum number of images to generate')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Maximum number of worker processes (None for auto)')
    parser.add_argument('--concurrent', action='store_true', default=True,
                        help='Use concurrent processing mode')
    parser.add_argument('--serial', dest='concurrent', action='store_false',
                        help='Use serial processing mode (disables concurrent)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Size of generated images (width and height)')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
        settings = yaml.safe_load(file)
    
    images_dir = args.input_dir
    output_dir = args.output_dir
    modality = args.modality
    MAX_IMAGES = args.max_images
    MAX_WORKERS = args.max_workers
    USE_CONCURRENT = args.concurrent
    IMAGE_SIZE = args.image_size
    
    print(f"Starting image processing, modality: {modality}")
    print(f"Input directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target generation count: {MAX_IMAGES}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Processing mode: {'Concurrent' if USE_CONCURRENT else 'Serial'}")
    if MAX_WORKERS:
        print(f"Worker processes: {MAX_WORKERS}")
    
    start_time = time.time()

    if USE_CONCURRENT:
        print("=== Using concurrent mode ===")
        successful, failed = process_images_concurrent(
            images_dir=images_dir,
            modality=modality,
            settings=settings,
            max_images=MAX_IMAGES,
            max_workers=MAX_WORKERS,
            output_dir=output_dir,
            image_size=IMAGE_SIZE
        )
        print(f"Concurrent processing result: Success {successful}, Failed {failed}")
        
    else:
        print("=== Using serial mode ===")
        count = 1
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        images = os.listdir(images_dir)
        
        with tqdm(total=MAX_IMAGES, desc="Processing images") as pbar:
            while count <= MAX_IMAGES:
                for image in images:
                    if count > MAX_IMAGES:
                        break
                        
                    image_path = os.path.join(images_dir, image)
                    network = Network(canvas, settings, modality=modality)
                    root = resetNetwork(network, canvas, settings, image_path, modality)
                    
                    if root is None:
                        continue
                    
                    iterationAmount = 0
                    last = 0
                    
                    while True:
                        network.update()
                        iterationAmount += 1
                        
                        if len(network.nodes) == last:
                            break
                        last = len(network.nodes)
                        
                        if iterationAmount > 10000:
                            break
                    
                    draw(root, count, modality=modality, output_dir=output_dir, image_size=IMAGE_SIZE)
                    count += 1
                    pbar.update(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
    print(f"Average time per image: {elapsed_time/MAX_IMAGES:.2f} seconds")
    
    if USE_CONCURRENT:
        print(f"Concurrent mode used {MAX_WORKERS or min(mp.cpu_count(), 4)} processes")
    
    print("Processing complete!")

if __name__ == "__main__":
    mp.freeze_support()
    main()