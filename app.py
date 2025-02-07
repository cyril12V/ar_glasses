import os
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from math import atan2, degrees

# Supprimer les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation Flask
app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app)

# Initialisation Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Variable globale pour stocker l'ID des lunettes
current_glasses_id = 1

# Constante pour les calculs (distance réelle entre les yeux en cm)
REAL_EYE_DISTANCE_CM = 6.5

def calculate_distance(point1, point2, image_width, image_height):
    x1, y1 = int(point1.x * image_width), int(point1.y * image_height)
    x2, y2 = int(point2.x * image_width), int(point2.y * image_height)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Fonction pour pivoter une image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

# Route principale
@app.route('/index')
def index():
    return render_template('index.html')

# Route principale
@app.route('/lunette1')
def lunette1():
    return render_template('lunettes1.html')

@app.route('/lunette2')
def lunette2():
    return render_template('lunettes2.html')

@app.route('/lunette3')
def lunette3():
    return render_template('lunettes3.html')

@app.route('/lunette4')
def lunette4():
    return render_template('lunettes4.html')
# Route principale
@app.route('/')
def lp():
    return render_template('lp.html')

# Route principale
@app.route('/catalogue')
def catalogue():
    return render_template('catalogue.html')

@app.route('/analyse-visage', methods=['GET'])
def analysevisagepage():
    return render_template('analyse-visage.html')  # Affiche la page HTML

@app.route('/api/analyse-visage', methods=['POST'])
def analyse_visage():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Impossible d'accéder à la caméra."})

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                chin = face_landmarks.landmark[152]
                forehead = face_landmarks.landmark[10]
                left_cheek = face_landmarks.landmark[234]
                right_cheek = face_landmarks.landmark[454]

                eye_distance_pixels = calculate_distance(left_eye, right_eye, w, h)
                face_height_pixels = calculate_distance(chin, forehead, w, h)
                face_width_pixels = calculate_distance(left_cheek, right_cheek, w, h)

                if eye_distance_pixels > 0:
                    face_height_cm = (face_height_pixels / eye_distance_pixels) * REAL_EYE_DISTANCE_CM
                    face_width_cm = (face_width_pixels / eye_distance_pixels) * REAL_EYE_DISTANCE_CM
                    eye_distance_cm = REAL_EYE_DISTANCE_CM
                else:
                    return jsonify({"error": "Landmarks introuvables."})

                face_shape = "Oval" if face_width_cm / face_height_cm > 0.8 else "Rectangulaire"

                cap.release()
                return jsonify({
                    "face_size": round(face_width_cm, 2),
                    "face_shape": face_shape,
                    "eye_distance": round(eye_distance_cm, 2)
                })

    cap.release()
    return jsonify({"error": "Aucun visage détecté."})

# WebSocket pour démarrer le flux vidéo
@socketio.on('start_video')
def start_video(data):
    global current_glasses_id
    current_glasses_id = data.get('glasses_id', 1)

    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                emit('error', {'message': "Impossible de lire la caméra."})
                break

            # Convertir en RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Charger les lunettes en fonction de l'ID actuel
            sunglasses_path = f"static/glasses{current_glasses_id}.png"
            sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)

            # Ajouter les lunettes si un visage est détecté
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    left_eye = face_landmarks.landmark[33]
                    right_eye = face_landmarks.landmark[263]

                    left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
                    right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

                    eye_center_x = (left_eye_x + right_eye_x) // 2
                    eye_center_y = (left_eye_y + right_eye_y) // 2
                    glasses_width = int(np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) * 2)
                    glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

                    dx = right_eye_x - left_eye_x
                    dy = right_eye_y - left_eye_y
                    angle = -degrees(atan2(dy, dx))

                    resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))
                    rotated_sunglasses = rotate_image(resized_sunglasses, angle)

                    for i in range(glasses_height):
                        for j in range(glasses_width):
                            y_offset = eye_center_y - glasses_height // 2 + i
                            x_offset = eye_center_x - glasses_width // 2 + j
                            if 0 <= y_offset < h and 0 <= x_offset < w and rotated_sunglasses[i, j, 3] != 0:
                                frame[y_offset, x_offset] = rotated_sunglasses[i, j, :3]

            # Convertir en JPEG et envoyer à la page web
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            emit('video_frame', {'image': frame_bytes}, namespace='/')

    cap.release()

# WebSocket pour mettre à jour les lunettes sélectionnées
@socketio.on('update_glasses')
def update_glasses(data):
    global current_glasses_id
    current_glasses_id = data.get('glasses_id', 1)
    emit('message', {'message': f'Lunettes {current_glasses_id} sélectionnées.'})

if __name__ == "__main__":
    print("Lien vers l'interface : http://127.0.0.1:5000")
    socketio.run(app, debug=True)