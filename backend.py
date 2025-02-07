import os
import cv2
import mediapipe as mp
import numpy as np
from math import atan2, degrees

# Supprimer les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation de Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Fonction pour pivoter une image
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

# Lire l'ID des lunettes sélectionnées depuis un fichier temporaire
def get_selected_glasses():
    with open("selected_glasses.txt", "r") as file:
        glasses_id = file.read().strip()
    return f"static/glasses{glasses_id}.png"

# Charger l'image des lunettes
sunglasses_path = get_selected_glasses()
sunglasses = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)  # Image avec transparence

# Initialisation de la caméra
cap = cv2.VideoCapture(0)

# Utiliser Face Mesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Impossible de lire la caméra.")
            break

        # Convertir l'image en RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Détecter les landmarks
        results = face_mesh.process(image)

        # Revenir à BGR pour OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Si des landmarks sont détectés
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Obtenir les coordonnées des yeux et du nez
                left_eye = face_landmarks.landmark[33]  # Œil gauche
                right_eye = face_landmarks.landmark[263]  # Œil droit
                nose = face_landmarks.landmark[1]  # Milieu du nez

                # Convertir les coordonnées normalisées en pixels
                h, w, _ = image.shape
                left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
                right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)

                # Calculer la position centrale et la taille des lunettes
                eye_center_x = (left_eye_x + right_eye_x) // 2
                eye_center_y = (left_eye_y + right_eye_y) // 2
                glasses_width = int(np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) * 2)
                glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

                # Calculer l'angle entre les yeux pour la rotation
                dx = right_eye_x - left_eye_x
                dy = right_eye_y - left_eye_y
                angle = -degrees(atan2(dy, dx))  # Angle d'inclinaison

                # Ajuster la position des lunettes en fonction du nez
                offset_x = nose_x - eye_center_x
                offset_y = int(nose_y - (eye_center_y + glasses_height * 0.3))

                # Redimensionner et pivoter les lunettes
                resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))
                rotated_sunglasses = rotate_image(resized_sunglasses, angle)

                # Ajouter les lunettes à l'image
                for i in range(glasses_height):
                    for j in range(glasses_width):
                        y_offset = eye_center_y + offset_y - glasses_height // 2 + i
                        x_offset = eye_center_x + offset_x - glasses_width // 2 + j
                        if y_offset >= h or x_offset >= w or y_offset < 0 or x_offset < 0:
                            continue
                        if rotated_sunglasses[i, j, 3] != 0:  # Si le pixel n'est pas transparent
                            image[y_offset, x_offset] = rotated_sunglasses[i, j, :3]

        # Afficher l'image
        cv2.imshow('Face with Sunglasses', image)

        # Quitter avec la touche 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
