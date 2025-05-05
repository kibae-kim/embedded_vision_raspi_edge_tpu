# movenet_lightning_inference.py
# CPU 기반 MoveNet SinglePose Lightning INT8 추론 코드

import cv2
import numpy as np
import time
import os
from tflite_runtime.interpreter import Interpreter

def get_abs_path(relative_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, relative_path.replace('/', os.sep))

MODEL_PATH = get_abs_path('models/cpu/movenet_singlepose_lightning_int8.tflite')
INPUT_SIZE = (192, 192)  # width, height
STRIDE = 4

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Cannot open webcam"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_image = cv2.resize(frame, INPUT_SIZE)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.uint8)
    input_data = np.expand_dims(input_image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = time.time()
    interpreter.invoke()
    elapsed = time.time() - start_time

    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    keypoints = keypoints_with_scores[0, 0, :, :]

    for idx, (y, x, score) in enumerate(keypoints):
        if score > 0.3:
            cx, cy = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    cv2.putText(frame, f"{elapsed * 1000:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow('MoveNet Lightning Pose Estimation (CPU)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
