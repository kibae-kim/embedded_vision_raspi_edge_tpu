import os
import cv2
import time
from pycoral.adapters import detect
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

def get_abs_path(relative_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # edge_demo 기준
    return os.path.join(root, relative_path.replace('/', os.sep))

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

def draw_objects(frame, objects):
    for obj in objects:
        bbox = obj.bbox
        label_text = f"ID:{obj.id} {obj.score:.2f}"
        cv2.rectangle(frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (bbox.xmin, bbox.ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame

def main():
    model_path = get_abs_path("models/edgetpu/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite")

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "USB Cannot be accessible"

    interpreter = load_model(model_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start = time.time()
        objs = run_inference(interpreter, frame)
        elapsed = time.time() - start
        frame = draw_objects(frame, objs)
        cv2.putText(frame, f"{elapsed * 1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("EdgeTPU Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
