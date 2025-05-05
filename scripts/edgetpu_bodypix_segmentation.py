# bodypix_segmentation_edgetpu.py - EdgeTPU 실시간 BodyPix 세그멘테이션
import os
import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input, output_tensor


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
    segmentation_map = output_tensor(interpreter, 0)
    return segmentation_map[0]


def decode_segmap(segmentation, num_classes=2):
    label_colors = np.array([[0, 0, 0], [0, 255, 0]], dtype=np.uint8)
    color_seg = label_colors[segmentation]
    return color_seg


def main():
    model_path = get_abs_path('models/edgetpu/bodypix_mobilenet_v1_075_768_576_16_quant_decoder_edgetpu.tflite')
    interpreter = load_model(model_path)

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "USB Camera is Not accessible"

    size = input_size(interpreter)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        seg = run_inference(interpreter, frame)
        elapsed = time.time() - start

        seg = cv2.resize(seg.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = decode_segmap(seg)
        blended = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

        cv2.putText(blended, f"{elapsed * 1000:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("BodyPix (EdgeTPU)", blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
