# scripts/cpu_bodypix_segmentation.py
import os
import cv2
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

def get_model_path(filename):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, 'models', 'cpu', filename)

def main():
    model_path = get_model_path('bodypix_mobilenet_v1_075_768_576_16_quant_decoder.tflite')
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "Camera not accessible"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (input_width, input_height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start = time.time()
        interpreter.invoke()
        inference_time = time.time() - start

        mask = interpreter.get_tensor(output_details[0]['index'])[0]
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_color = (mask_resized * 255).astype(np.uint8)
        mask_color = cv2.applyColorMap(mask_color, cv2.COLORMAP_JET)

        blended = cv2.addWeighted(frame, 0.5, mask_color, 0.5, 0)
        cv2.putText(blended, f"{inference_time * 1000:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('BodyPix CPU', blended)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

