# cpu_ssd_inference.py
import os
import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'models', 'cpu', 'tf2_ssd_mobilenet_v2_coco17_ptq.tflite')

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "USB Camera is Not accessible."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image_resized = cv2.resize(frame, (input_width, input_height))
    input_tensor = np.expand_dims(image_resized, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    start = time.time()
    interpreter.invoke()
    inference_time = time.time() - start

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [N,4]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # [N]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # [N]
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])  # scalar

    for i in range(count):
        if scores[i] >= 0.5:
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"ID:{int(classes[i])} {scores[i]:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f"{inference_time*1000:.1f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("CPU SSD Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
