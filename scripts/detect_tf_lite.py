import os
from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input, output_tensor

# 모델 경로 설정
HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(HERE, '..', 'models',
                   'ssd_mobilenet_v1_coco_ptq_edgetpu.tflite'))

# Edge TPU 전용 인터프리터 생성
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# 이미지 로드
img = Image.open('test.jpg').convert('RGB')
# 입력 전처리 예시 (모델 요구 크기로 리사이즈)
resized = img.resize((300, 300))
set_input(interpreter, np.asarray(resized).reshape(1,300,300,3))

interpreter.invoke()

# 결과 추출
boxes = output_tensor(interpreter, 0)
classes = output_tensor(interpreter, 1)
scores = output_tensor(interpreter, 2)
# …후처리 로직…
