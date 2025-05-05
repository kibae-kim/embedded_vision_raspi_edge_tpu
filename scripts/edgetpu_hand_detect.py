import cv2
import time
from pycoral.adapters import detect, common
from pycoral.utils.edgetpu import make_interpreter

def load_model(model_path):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, frame):
    input_size = common.input_size(interpreter)
    resized = cv2.resize(frame, input_size)
    common.set_input(interpreter, resized)
    interpreter.invoke()
    return detect.get_objects(interpreter, score_threshold=0.5)

def draw_boxes(frame, objects):
    for obj in objects:
        bbox = obj.bbox
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (255, 0, 0), 2)
        cv2.putText(frame, f"ID:{obj.id} {obj.score:.2f}",
                    (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return frame

def main():
    model_path = "models/edgetpu/hand_detection_full_integer_quant_edgetpu.tflite"
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "카메라를 열 수 없음"

    interpreter = load_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        objects = run_inference(interpreter, frame)
        elapsed = time.time() - start
        frame = draw_boxes(frame, objects)
        cv2.putText(frame, f"{elapsed * 1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
