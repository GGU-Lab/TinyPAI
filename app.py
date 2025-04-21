from flask import Flask, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import os
import subprocess

app = Flask(__name__)

# 경로 설정
IMAGE_NAME = "test.jpg"
IMAGE_PATH = f"/home/tglab/test1_project/{IMAGE_NAME}"
MODEL_PATH = os.path.join("models", "mobilenet_v1_1.0_224_quant.tflite")
LABEL_PATH = os.path.join("models", "labels.txt")

# 라벨 로드
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# TFLite 인터프리터 초기화
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 카메라 촬영
        env = os.environ.copy()
        # 라즈베리파이 GUI에서 테스트할 때는 --nopreview 옵션이 있어야 함 (cmd 디스플레이 없이 카메라 동작하게하기)
        subprocess.run(["libcamera-jpeg", "--nopreview", "-o", IMAGE_PATH], check=True)

        # 이미지 로드
        img = Image.open(IMAGE_PATH).convert('RGB')
        img = img.resize((224, 224))
        input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

        # 추론
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        end = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        num_classes = output_data.shape[1]
        print(f"모델 출력 클래스 수: {num_classes}, 라벨 개수: {len(labels)}")

        top_result = np.argmax(output_data)
        confidence = float(output_data[0][top_result]) / 255.0
        label = labels[top_result]

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 4),
            'elapsed_ms': round((end - start) * 1000, 2),
            'image_url': '/image'
        })

    except subprocess.CalledProcessError:
        return jsonify({'error': '카메라 촬영에 실패했습니다'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 정적 이미지 라우팅
@app.route('/image')
def get_image():
    return send_from_directory('/home/tglab/test1_project', IMAGE_NAME)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  