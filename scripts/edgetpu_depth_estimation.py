import cv2
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input

def load_model(model_path):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, frame):
    ih, iw = input_size(interpreter)
    resized = cv2.resize(frame, (iw, ih))
    set_input(interpreter, resized)
    interpreter.invoke()
    output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
    return output  # shape: [H, W] or [1, H, W, 1]

def normalize_depth(depth_map):
    depth_map = depth_map.squeeze()
    depth_map = np.clip(depth_map, 0, 255)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_map = (depth_map * 255).astype(np.uint8)
    return depth_map

def main():
    model_path = "models/edgetpu/depth_estimation_full_integer_quant_edgetpu.tflite"
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "카메라를 열 수 없음"

    interpreter = load_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        depth_raw = run_inference(interpreter, frame)
        elapsed = time.time() - start

        depth_map = normalize_depth(depth_raw)
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

        cv2.putText(depth_colored, f"{elapsed*1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Depth Estimation", depth_colored)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
