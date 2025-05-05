# edgetpu_bodypix_segmentation.py - EdgeTPU Real-Time BodyPix Segmentation (Decoder-Free)
import os
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input, output_tensor


def load_model(model_path):
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def segment_body(interpreter, frame):
    w, h = input_size(interpreter)
    resized = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    set_input(interpreter, rgb)
    interpreter.invoke()
    segmentation = output_tensor(interpreter, 0)
    segmentation = segmentation.squeeze()  # shape: (h, w)
    return segmentation


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, "models", "edgetpu", "bodypix_mobilenet_v1_075_768_576_16_quant_decoder_edgetpu.tflite")

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Webcam not accessible"

    interpreter = load_model(model_path)
    w, h = input_size(interpreter)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = cv2.getTickCount()
        mask = segment_body(interpreter, frame)
        mask_resized = cv2.resize(mask.astype(np.uint8)*255, (frame.shape[1], frame.shape[0]))
        mask_color = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, mask_color, 0.4, 0)

        time_ms = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        cv2.putText(overlay, f"{time_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("EdgeTPU BodyPix", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
