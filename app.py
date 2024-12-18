import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

# Flask-App initialisieren
app = Flask(__name__)

# Verzeichnis für Uploads und erlaubte Dateiformate festlegen
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

# Lade den Gesichtserkennungsmodell
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funktion zur Überprüfung der Dateiendung
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Funktion für Face-Swap
def face_swap(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    faces1 = detector(gray1)
    faces2 = detector(gray2)

    if len(faces1) == 0 or len(faces2) == 0:
        print("Kein Gesicht gefunden!")
        return None

    face1 = faces1[0]
    face2 = faces2[0]

    landmarks1 = predictor(gray1, face1)
    landmarks2 = predictor(gray2, face2)

    points1 = np.array([(p.x, p.y) for p in landmarks1.parts()])
    points2 = np.array([(p.x, p.y) for p in landmarks2.parts()])

    mask1 = np.zeros_like(gray1)
    mask2 = np.zeros_like(gray2)

    cv2.fillConvexPoly(mask1, points1, 255)
    cv2.fillConvexPoly(mask2, points2, 255)

    face1_img = cv2.bitwise_and(image1, image1, mask=mask1)
    face2_img = cv2.bitwise_and(image2, image2, mask=mask2)

    return face1_img, face2_img

# Startseite (Formular für den Bild-Upload)
@app.route('/')
def index():
    return render_template('index.html')

# Route für den Upload und Face-Swap-Prozess
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file1' not in request.files or 'file2' not in request.files:
        return redirect(request.url)

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # Bilder laden
        img1 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        img2 = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # Face-Swap durchführen
        face1_img, face2_img = face_swap(img1, img2)

        # Speichern des Ergebnisses
        result_filename = 'result.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)

        if face1_img is not None and face2_img is not None:
            combined = np.hstack([face1_img, face2_img])  # Gesichter nebeneinander anzeigen
            cv2.imwrite(result_path, combined)

            return render_template('result.html', result_image=result_filename)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
