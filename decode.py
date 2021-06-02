import numpy as np
from scipy.ndimage import maximum_filter


def decode_centers(output, score_threshold=0.4, max_detections=100):
    batch_centers = []
    for i in range(output.shape[0]):
        max_pool = maximum_filter(output[i, :, :], size=5, mode='constant', cval=0.0)
        filtered_heatmap = np.where(max_pool == output[i, :, :], output[i, :, :], 0.0)
        filtered_heatmap[filtered_heatmap < score_threshold] = 0.0

        centers = top_k(filtered_heatmap, max_detections)

        filtered_centers = []
        for center in centers:
            if center[0] > 0:
                filtered_centers.append(center.tolist())
        filtered_centers.sort(key=lambda x: x[1])

        batch_centers.append(filtered_centers)

    return batch_centers


def decode_centers_and_scales(output, score_threshold, max_detections,):
    batch_centers = decode_centers(output[:, :, :, :1], score_threshold, max_detections)

    scales_array = output[:, :, :, 1:]

    batch_detections = []

    for i in range(output.shape[0]):
        centers = batch_centers[i]
        detections = []

        for center in centers:
            if len(center) != 0:
                detection = center + [scales_array[i, int(center[2]), int(center[1]), 0]]
                detection = detection + [scales_array[i, int(center[2]), int(center[1]), 1]]

                detections.append(detection)

        batch_detections.append(detections)

    return batch_detections


def top_k(array, k):
    scores = np.full((k,), -1.0, dtype=np.float32)
    x = np.full((k,), -1.0, dtype=np.float32)
    y = np.full((k,), -1.0, dtype=np.float32)

    tmp = array.flatten()
    for i in range(k):
        idx = tmp.argmax()
        scores[i] = tmp[idx]
        y[i], x[i] = np.unravel_index(idx, array.shape)
        tmp[idx] = -1.0

    top_array = np.stack([scores, x, y], axis=-1)
    return top_array
