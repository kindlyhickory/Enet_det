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

        for a in range(len(filtered_centers)):
            if len(filtered_centers[a] == 0):
                continue

            for b in range(len(filtered_centers)):
                if len(filtered_centers[b] == 0):
                    continue

                dx = abs(filtered_centers[a][1] - filtered_centers[b][1])
                dy = abs(filtered_centers[a][2] - filtered_centers[b][2])
                if dx < 10 and dy < 10 and filtered_centers[a] != filtered_centers[b]:
                    if filtered_centers[a][0] > filtered_centers[b][0]:
                        filtered_centers[b] = []
                    else:
                        filtered_centers[a] = []

        batch_centers.append(filtered_centers)

    return batch_centers


def decode_centers_and_scales(output, score_threshold=0.4, max_detections=100):
    batch_centers = decode_centers(output[:, :, :], score_threshold, max_detections)

    # scales_array = output[:, :, :]

    batch_detections = []

    for i in range(output.shape[0]):
        centers = batch_centers[i]
        detections = []

        for center in centers:
            if len(center) != 0:
                detection = center + [output[i, int(center[2]), int(center[1])]]
                detection = detection + [output[i, int(center[2]), int(center[1])]]

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
