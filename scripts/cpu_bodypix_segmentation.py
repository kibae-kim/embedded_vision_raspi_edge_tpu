# cpu_bodypix_segmentation.py - CPU 기반 BodyPix 세분화
import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# 절대 경로 설정
def get_model_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "cpu", "bodypix_mobilenet_v1_075_768_576_16_quant_decoder.tflite")

# 입력 프레임으로부터 마스크 예측
def run_segmentation(interpreter, frame):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h, w = input_details[0]['shape'][1:3]

    resized = cv2.resize(frame, (w, h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(rgb.astype(np.uint8), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]  # (h, w)
    return output

# 메인 루프
def main():
    model_path = get_model_path()
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), "카메라 접근 실패"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = run_segmentation(interpreter, frame)
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(frame, 0.6, mask_color, 0.4, 0)

        cv2.imshow("CPU BodyPix", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
