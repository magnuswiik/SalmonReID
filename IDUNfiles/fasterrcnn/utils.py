import numpy as np
import matplotlib.pyplot as plt
import cv2
import os, json

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = intersection / float(area_box1 + area_box2 - intersection)
    return iou

def visualize_annot(imagepath):

    image = cv2.cvtColor(cv2.imread(imagepath), cv2.COLOR_BGR2RGB)
    annotpath = imagepath[:-3] + "json"

    boxes = []

    with open(annotpath,'r') as file:
        content = json.load(file)
        shapes = content['shapes']
        for shape in shapes:
            boxes.append(shape['points'])

    # Draw bounding boxes on the image
    for box in boxes:
        min, max = box
        cv2.rectangle(image, (int(min[0]), int(min[1])), (int(max[0]), int(max[1])), (0, 255, 0), 2)

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def visualize_preds(task, image, predictions):
    
    if task == "salmon":
        labels_map = {"background": 0, "salmon":1}
    if task == "landmarks":
        labels_map = {"background": 0, 'tailfin': 1, 'dorsalfin': 2, 'thorax': 3, 'pectoralfin': 4, 'eyeregion': 5, 'pectoral': 4} # Husk å fjerne pectoral og fikse datasett!

    image = image.cpu().numpy().transpose((1, 2, 0))
    image = np.array(image * 255, dtype=np.uint8)
    boxes = predictions['boxes'].detach().cpu().numpy()
    labels = predictions['labels'].detach().cpu().numpy()
    scores = predictions['scores'].detach().cpu().numpy()

    for j in range(len(boxes)):
        x1, y1, x2, y2 = boxes[j]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        class_label = list(labels_map.keys())[list(labels_map.values()).index(labels[j])]
        text = f'{class_label}: {scores[j]:.2f}'
        cv2.putText(image, text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def pred_to_labelme(pred, imagepath):
    
    pred = pred[0] # Remove batch dimension
    
    labelme_dict = {'version':'5.3.1',
                'flags':{},
                'shapes':[],
                'imagePath':'',
                'imageData': None,
                'imageHeight':1440,
                'imageWidth':1920
                }
    
    filename= imagepath[:-3] + 'json'
    
    pred_labels = [f"salmon: {score:.3f}" for score in pred["scores"]]
    pred_boxes = pred["boxes"].tolist()
    
    with open(filename, 'w') as file:
        labelme_dict['imagePath'] = imagepath
        shapes_list = []
        
        for i in range(len(pred_labels)):
            shape_dict = {"label": "salmon", "points": [], 'group_id': None, 'description': '', 'shape_type': 'rectangle', 'flags': {}}
            shapes_list.append(shape_dict)

        for i in range(len(pred_boxes)):
            box = pred_boxes[i]
            shape_dict = shapes_list[i]
            shape_dict['points'] = [box[:2], box[2:]]
        
        labelme_dict['shapes'] = shapes_list
        json.dump(labelme_dict, file, indent=2)
