# posenet_resnet50_edgetpu.py - EdgeTPU Real Time Pose Estimation
import os
import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input, output_tensor

# COCO Pose Keypoint hue and connection 
def draw_pose(frame, keypoints, threshold=0.2):
    COCO_EDGES = [
        (0, 1), (1, 3), (0, 2), (2, 4),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for i, (y, x, score) in enumerate(keypoints):
        if score > threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)
    for a, b in COCO_EDGES:
        if keypoints[a][2] > threshold and keypoints[b][2] > threshold:
            pt1 = (int(keypoints[a][1]), int(keypoints[a][0]))
            pt2 = (int(keypoints[b][1]), int(keypoints[b][0]))
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    return frame


def get_abs_path(relative_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, relative_path.replace('/', os.sep))


def load_model(model_path):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, frame):
    size = input_size(interpreter)
    image = cv2.resize(frame, size)
    set_input(interpreter, image)
    interpreter.invoke()
    heatmaps = output_tensor(interpreter, 0)[0]  # [H, W, 17]
    offsets = output_tensor(interpreter, 1)[0]   # [H, W, 34]
    return heatmaps, offsets


def decode_pose(heatmaps, offsets, stride=16):
    num_keypoints = heatmaps.shape[-1]
    keypoints = []
    for i in range(num_keypoints):
        hmap = heatmaps[:, :, i]
        y, x = np.unravel_index(np.argmax(hmap), hmap.shape)
        score = hmap[y, x]
        offset_y = offsets[y, x, i]
        offset_x = offsets[y, x, i + num_keypoints]
        keypoints.append((y * stride + offset_y, x * stride + offset_x, score))
    return keypoints


def main():
    model_path = get_abs_path('models/edgetpu/posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite')
    interpreter = load_model(model_path)
    size = input_size(interpreter)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "USB Camera is Not Accessible"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        heatmaps, offsets = run_inference(interpreter, frame)
        keypoints = decode_pose(heatmaps, offsets)
        elapsed = time.time() - start

        annotated = draw_pose(frame.copy(), keypoints)
        cv2.putText(annotated, f"{elapsed * 1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("PoseNet (EdgeTPU)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
