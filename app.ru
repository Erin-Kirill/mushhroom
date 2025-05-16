from flask import Flask, request, render_template, g
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.config['DATABASE'] = 'mushroom_predictions.db'  # Путь к базе данных
model = load_model('mushroom_classifier_model_final.h5')
class_names = sorted(os.listdir('train_data'))

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row  # Для доступа к полям по имени
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()

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

                # Сохранение в БД
                db = get_db()
                db.execute(
                    'INSERT INTO predictions (filename, predicted_class, confidence) VALUES (?, ?, ?)',
                    (file.filename, prediction, confidence)
                )
                db.commit()

    db = get_db()
    history = db.execute(
        'SELECT * FROM predictions ORDER BY uploaded_at DESC LIMIT 5'
    ).fetchall()

    return render_template('index.html',
                         prediction=prediction,
                         confidence=confidence,
                         image_path=image_path,
                         history=history)

if __name__ == '__main__':
    init_db()  # Инициализация БД при старте
    app.run(debug=True)
