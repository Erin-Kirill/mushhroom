from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('mushroom_classifier_model_final.h5')

# Загружаем список классов из структуры папок или вручную
class_names = sorted(os.listdir('train_data'))  # или вручную: ['Белый гриб', 'Мухомор', ...]

def prepare_image(path):
    img = image.load_img(path, target_size=(160, 160))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                save_folder = os.path.join('static', 'uploads')
                os.makedirs(save_folder, exist_ok=True)
                filepath = os.path.join(save_folder, file.filename)
                file.save(filepath)

                img = prepare_image(filepath)
                pred = model.predict(img)[0]
                predicted_class = class_names[np.argmax(pred)]
                confidence = round(np.max(pred) * 100, 2)
                prediction = predicted_class
                image_path = '/' + filepath.replace('\\', '/')

    return render_template('index.html',
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
