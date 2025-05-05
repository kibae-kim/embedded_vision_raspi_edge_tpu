# cpu_posenet_resnet50.py - CPU용 PoseNet (Decoder 없이 직접 출력 처리)
import os
import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

# 절대 경로 구성
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'cpu', 'posenet_resnet_50_416_288_16_quant_cpu_decoder.tflite')

# 입력 크기 및 키포인트 수 정의
INPUT_SIZE = (416, 288)
NUM_KEYPOINTS = 17
KEYPOINT_EDGES = [  # COCO 포맷
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]

# 모델 로딩 및 입력/출력 텐서 초기화
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 웹캠 열기
cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Camera not accessible"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_img = cv2.resize(frame, INPUT_SIZE)
    input_tensor = np.expand_dims(input_img.astype(np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    start = time.time()
    interpreter.invoke()
    elapsed = (time.time() - start) * 1000

    # 출력 해석
    heatmap = interpreter.get_tensor(output_details[0]['index'])[0]  # (H/stride, W/stride, 17)
    offsets = interpreter.get_tensor(output_details[1]['index'])[0]  # (H/stride, W/stride, 34)

    stride = frame.shape[1] // heatmap.shape[1]
    keypoints = []

    for i in range(NUM_KEYPOINTS):
        heatmap_i = heatmap[:, :, i]
        y, x = np.unravel_index(np.argmax(heatmap_i), heatmap_i.shape)
        offset_y = offsets[y, x, i]
        offset_x = offsets[y, x, i + NUM_KEYPOINTS]
        kp_x = (x * stride + offset_x) * frame.shape[1] / INPUT_SIZE[0]
        kp_y = (y * stride + offset_y) * frame.shape[0] / INPUT_SIZE[1]
        keypoints.append((int(kp_x), int(kp_y)))

    # 시각화
    for x, y in keypoints:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in KEYPOINT_EDGES:
        cv2.line(frame, keypoints[a], keypoints[b], (255, 0, 0), 2)

    cv2.putText(frame, f"{elapsed:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Pose Estimation (CPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
