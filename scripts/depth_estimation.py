import os
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input, output_tensor

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.abspath(os.path.join(HERE, '..', 'models',
                   'midas_v2_small_edgetpu.tflite'))

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

img = Image.open('scene.jpg').convert('RGB')
# MiDaS small은 256×256 입력이라 가정
inp = img.resize((256,256))
set_input(interpreter, np.expand_dims(inp, 0))

interpreter.invoke()

depth_map = output_tensor(interpreter, 0)  # shape e.g. (1,256,256,1)
depth_map = depth_map.squeeze()
# depth_map 후처리·시각화…
