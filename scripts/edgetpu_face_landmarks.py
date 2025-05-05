import cv2
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input, input_size

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
    landmarks = np.reshape(output, (-1, 3))  # (468, [x, y, score])
    h, w, _ = frame.shape
    landmarks[:, 0] *= w  # x
    landmarks[:, 1] *= h  # y
    return landmarks

def draw_landmarks(frame, landmarks, score_threshold=0.5):
    for x, y, score in landmarks:
        if score > score_threshold:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
    return frame

def main():
    model_path = "models/edgetpu/face_landmarks_detector_1x3x256x256_full_integer_quant_edgetpu.tflite"
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "카메라를 열 수 없음"

    interpreter = load_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        landmarks = run_inference(interpreter, frame)
        elapsed = time.time() - start
        frame = draw_landmarks(frame, landmarks)
        cv2.putText(frame, f"{elapsed*1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Face Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
