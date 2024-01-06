import numpy as np

def calculate_iou(pred_frames, gt_frames):
    # Reshape frames to 1D arrays
    pred_frames_flat = pred_frames.reshape(-1, 224 * 224 * 3)
    gt_frames_flat = gt_frames.reshape(-1, 224 * 224 * 3)

    # Convert to binary masks
    pred_masks = (pred_frames_flat > 0).astype(int)
    gt_masks = (gt_frames_flat > 0).astype(int)

    # Calculate intersection and union
    intersection = np.sum(np.logical_and(pred_masks[:, None, :], gt_masks), axis=(1, 2))
    union = np.sum(np.logical_or(pred_masks[:, None, :], gt_masks), axis=(1, 2))

    # Calculate IoU
    iou = intersection / union

    # Return the mean IoU across all frames
    mean_iou = np.mean(iou)

    return mean_iou

# Example usage
pred_keyframes = np.random.randint(0, 2, size=(300, 224, 224, 3))
gt_frames = np.random.randint(0, 2, size=(21, 224, 224, 3))

iou_metric = calculate_iou(pred_keyframes, gt_frames)
print("Intersection over Union (IoU) Metric:", iou_metric)
