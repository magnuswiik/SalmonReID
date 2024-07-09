import numpy as np
import matplotlib.pyplot as plt

# Visualize model predictions
def visualize_preds(task, image, predictions, targets):
    
    if task == "salmon":
        labels_map = {"background": 0, "salmon":1}
    if task == "bodyparts":
        labels_map = {"background": 0, 'tailfin': 1, 'dorsalfin': 2, 'thorax': 3, 'pectoralfin': 4, 'eyeregion': 5} # Husk å fjerne pectoral og fikse datasett!

    image = image.cpu().numpy().transpose((1, 2, 0))
    image = np.array(image * 255, dtype=np.uint8)
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    
    targ_boxes = targets['boxes'].detach().cpu().numpy()
    targ_labels = targets['labels'].detach().cpu().numpy()
    
    pred_boxes = predictions['boxes'].detach().cpu().numpy()
    pred_labels = predictions['labels'].detach().cpu().numpy()
    pred_scores = predictions['scores'].detach().cpu().numpy()

    for pred_box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        
        x1, y1, x2, y2 = pred_box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        class_label = list(labels_map.keys())[list(labels_map.values()).index(label)]
        ax.text(x2 + 10, y1 + (y2-y1)/2, f'{class_label}:{round(score.item(), 3)}', color='white', fontsize=12, fontweight='bold')
        
    for targ_box in targ_boxes:
        
        x1, y1, x2, y2 = targ_box
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='lightgreen', facecolor='none')
        ax.add_patch(rect)

    plt.show()