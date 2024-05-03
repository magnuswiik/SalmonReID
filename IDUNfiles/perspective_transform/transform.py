import torch
import cv2
import numpy as np
import os, json, math
import matplotlib.pyplot as plt
import pandas as pd

def calculate_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cos_angle = dot_product / norm_product
    angle = np.arccos(cos_angle)
    return np.degrees(angle)

def is_singular(src_points, dst_points, warped_image, warped_points):
    # Check if source points are collinear
    if np.linalg.matrix_rank(np.transpose(src_points)) < 2:
        print("Points are collinear!")
        return True
    
    # Check if transformation results in points going to infinity
    if np.any(np.isinf(dst_points)):
        print("Points go to infinity!")
        return True
    
    threshold_angle = 0.5
    # Check if any two lines are parallel or nearly parallel (angle close to 180 degrees)
    for i in range(len(src_points) - 1):
        for j in range(i + 1, len(src_points)):
            line1 = src_points[i] - src_points[i - 1]
            line2 = src_points[j] - src_points[j - 1]
            angle = np.arccos(np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2)))
            if np.abs(np.degrees(angle) - 180) < threshold_angle:
                print("Lines are parallell!")
                visualize_transformed_image(warped_image, warped_points) 
                return True
    
    return False

def perspective_transform(image1, image2, from_keypoints, to_keypoints, boxes):
    
    # Find homography
    h, mask = cv2.findHomography(from_keypoints, to_keypoints, cv2.RANSAC)
    print("Homography matrix:")
    print(h)

    # Use homography
    height, width, channels = image2.shape
    img2Reg = cv2.warpPerspective(image1, h, (width, height))
    print("Warped image dimensions:", img2Reg.shape)

    transformed_points = []

    # Apply homogeneous transformation
    for points in boxes:
        point1 = points[:3]
        point2 = points[3:]
        transformed_point1 = np.dot(h, point1)
        transformed_point2 = np.dot(h, point2)
        normalized_point1 = transformed_point1[:2]/transformed_point1[2]
        normalized_point2 = transformed_point2[:2]/transformed_point2[2]
        transformed_points.append([normalized_point1, normalized_point2])
        
        singular = is_singular(from_keypoints, to_keypoints, img2Reg, transformed_points)
        
    return img2Reg, transformed_points

def visualize_transformed_image(image, points):
    
    for box in points:
        min, max = box
        if not math.isnan(min[0]) and not math.isnan(min[1]) and not math.isnan(max[0]) and not math.isnan(max[1]):
            cv2.rectangle(image, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (0, 0, 255), 2)
        else: 
            print('A transformed point is nan...')
            

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
def get_images_and_annots(datapath):
    annots = []
    imgs = []

    for folder in os.listdir(datapath):
        if (not folder.startswith('.')) and os.path.isdir(os.path.join(datapath, folder)):
            for file in sorted(os.listdir(os.path.join(datapath, folder))):
                if file.endswith('.json'):
                    file = os.path.join(datapath, file.split('_')[0], file)
                    annots.append(file)
            for file in sorted(os.listdir(os.path.join(datapath, folder))):
                file = os.path.join(datapath, file.split('_')[0], file)
                if file.endswith(('.jpg', '.jpeg', '.png')) and file[:-4] + '.json' in annots:
                    imgs.append(file)
    
    return imgs, annots


def main():
    
    from_path = "/Users/magnuswiik/prosjektoppgave_data/Masteroppgave_data/Helfisk_Landmark_Deteksjonssett/"

    imgs, annots = get_images_and_annots(from_path)
    
    # Warp every image into this image
    reference_image_path = os.path.join(from_path, "fish9/fish9_GP020101_00005919.jpg")
    reference_image = cv2.cvtColor(cv2.imread(reference_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    reference_annot_path = os.path.join(from_path, "fish9/fish9_GP020101_00005919.json")
    reference_points = np.array([[488.179168, 69.288451],[627.979469, 49.451411],
                                 [834.962545, 286.265656],[29.634321, 307.930108]], dtype=np.float32)
    reference_points = np.zeros((4, 2), dtype=np.float32)
    reference_boxes = np.zeros((5, 6), dtype=np.float32)
    
    with open(reference_annot_path, 'r') as file:
        content = json.load(file)
        shapes = content['shapes']
        
        index_point = 0 
        index_box = 0 
        for shape in shapes:
            if shape['shape_type'] == 'point':
                if shape['label'] == 'dorsalback' or shape['label'] == 'dorsalfront' or shape['label'] == 'pectoral' or shape['label'] == 'tailbot':
                    point = shape['points']
                    print(shape['label'])
                    x1 = float(point[0][0])
                    y1 = float(point[0][1])
                    reference_points[index_point] = [x1, y1]
                    index_point += 1
                
            if shape['shape_type'] == 'rectangle':
                box = shape['points']
                x1 = float(box[0][0])
                y1 = float(box[0][1])
                x2 = float(box[1][0])
                y2 = float(box[1][1])
                reference_boxes[index_box] = [x1, y1, 1, x2, y2, 1]
                index_box += 1
                
    '''# Add points to image
    for point in reference_points:
        cv2.circle(reference_image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    # Add boxes to image
    for box in reference_boxes:
        min, max = box[:3], box[3:]
        cv2.rectangle(reference_image, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (0, 0, 255), 2)
    
    plt.imshow(reference_image)
    plt.axis('off')
    plt.show()'''
    
    all_points = {
        'dorsalbackx': [],
        'dorsalbacky': [],
        'dorsalfrontx': [],
        'dorsalfronty': [],
        'pectoralx': [],
        'pectoraly': [],
        'tailbotx': [],
        'tailboty': []
    }

    for i in range(len(imgs)):
        
        image = cv2.cvtColor(cv2.imread(imgs[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Extract location of good matches
        points = np.zeros((4, 2), dtype=np.float32)
        boxes = np.zeros((5, 6), dtype=np.float32)

        index_point = 0 
        index_box = 0 
        with open(annots[i], 'r') as file:
            content = json.load(file)
            shapes = content['shapes']
            
            shapes_left = ['dorsalback','dorsalfront','pectoral','tailbot']
            
            for shape in shapes:
                if shape['shape_type'] == 'point':
                    if shape['label'] == 'dorsalback' or shape['label'] == 'dorsalfront' or shape['label'] == 'pectoral' or shape['label'] == 'tailbot':
                        shapes_left.remove(shape['label'])
                        point = shape['points']
                        x1 = float(point[0][0])
                        y1 = float(point[0][1])
                        points[index_point] = [x1, y1]
                        index_point += 1
                        all_points[shape['label']+'x'].append(x1)
                        all_points[shape['label']+'y'].append(y1)
                    
                if shape['shape_type'] == 'rectangle':
                    box = shape['points']
                    x1 = float(box[0][0])
                    y1 = float(box[0][1])
                    x2 = float(box[1][0])
                    y2 = float(box[1][1])
                    boxes[index_box] = [x1, y1, 1, x2, y2, 1]
                    index_box += 1
                
            if len(shapes_left) != 0:
                print('shape is missing!')
        
        warped_image, warped_points = perspective_transform(image, reference_image, points, reference_points, boxes)
        visualize_transformed_image(warped_image, warped_points) 
    
    avg_point = {
        'dorsalbackx': 0,
        'dorsalbacky': 0,
        'dorsalfrontx': 0,
        'dorsalfronty': 0,
        'pectoralx': 0,
        'pectoraly': 0,
        'tailbotx': 0,
        'tailboty': 0
    }
            
    for label, points in all_points.items():
        for p in points:
            avg_point[label] += p
        avg_point[label] /= len(points)
    
    print(avg_point)
    df = pd.DataFrame(all_points)
    print(df.median())
            
            
    '''# Add points to image
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

        # Add boxes to image
        for box in boxes:
            min, max = box[:3], box[3:]
            cv2.rectangle(image, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (0, 0, 255), 2)
        
        plt.imshow(image)
        plt.axis('off')
        plt.show()'''      
        

if __name__ == '__main__':
    main()
    
    