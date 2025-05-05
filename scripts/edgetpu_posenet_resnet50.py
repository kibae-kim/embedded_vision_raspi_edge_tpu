# edgetpu_posenet_resnet50.py - EdgeTPU ResNet50 기반 실시간 포즈 추정
import os
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input, output_tensor

KEYPOINT_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

def get_model_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "edgetpu", "posenet_resnet_50_416_288_16_quant_edgetpu_decoder.tflite")


def decode_pose(heatmap, offsets, stride, input_dims, orig_dims):
    h, w, num_kp = heatmap.shape
    kp_coords = []

    for i in range(num_kp):
        heatmap_i = heatmap[:, :, i]
        y, x = np.unravel_index(np.argmax(heatmap_i), heatmap_i.shape)
        offset_y = offsets[y, x, i]
        offset_x = offsets[y, x, i + num_kp]
        kp_x = (x * stride + offset_x) * orig_dims[0] / input_dims[0]
        kp_y = (y * stride + offset_y) * orig_dims[1] / input_dims[1]
        kp_coords.append((int(kp_x), int(kp_y)))

    return kp_coords


def main():
    model_path = get_model_path()
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    input_w, input_h = input_size(interpreter)
    stride = 16

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Camera cannot be opened."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (input_w, input_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        set_input(interpreter, rgb)
        interpreter.invoke()

        heatmap = output_tensor(interpreter, 0).squeeze()  # (H, W, 17)
        offsets = output_tensor(interpreter, 1).squeeze()  # (H, W, 34)
        keypoints = decode_pose(heatmap, offsets, stride, (input_w, input_h), (frame.shape[1], frame.shape[0]))

        for x, y in keypoints:
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        for a, b in KEYPOINT_EDGES:
            cv2.line(frame, keypoints[a], keypoints[b], (255, 0, 0), 2)

        cv2.imshow("EdgeTPU PoseNet", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
