import os
import cv2
import mediapipe as mp
import numpy as np

# Supprimer les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation de Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Charger l'image des lunettes
sunglasses = cv2.imread('photolunette.png', cv2.IMREAD_UNCHANGED)  # Image avec transparence

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
                # Obtenir les coordonnées des yeux (landmarks spécifiques)
                left_eye = face_landmarks.landmark[33]  # Œil gauche
                right_eye = face_landmarks.landmark[263]  # Œil droit

                # Convertir les coordonnées normalisées en pixels
                h, w, _ = image.shape
                left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
                right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

                # Calculer la position et la taille des lunettes
                eye_center_x = (left_eye_x + right_eye_x) // 2
                eye_center_y = (left_eye_y + right_eye_y) // 2
                glasses_width = int(np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) * 2)
                glasses_height = int(glasses_width * sunglasses.shape[0] / sunglasses.shape[1])

                # Redimensionner les lunettes
                resized_sunglasses = cv2.resize(sunglasses, (glasses_width, glasses_height))

                # Ajouter les lunettes à l'image (fusion alpha)
                for i in range(glasses_height):
                    for j in range(glasses_width):
                        if eye_center_y - glasses_height // 2 + i >= h or eye_center_x - glasses_width // 2 + j >= w:
                            continue
                        if resized_sunglasses[i, j, 3] != 0:  # Si le pixel n'est pas transparent
                            image[eye_center_y - glasses_height // 2 + i,
                                  eye_center_x - glasses_width // 2 + j] = resized_sunglasses[i, j, :3]

        # Afficher l'image
        cv2.imshow('Face with Sunglasses', image)

        # Quitter avec la touche 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
