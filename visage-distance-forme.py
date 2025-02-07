import os
import cv2
import mediapipe as mp
import math

# Supprimer les avertissements TensorFlow et Mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation de Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Facteur d'échelle (distance réelle entre les yeux en cm)
REAL_EYE_DISTANCE_CM = 6.5  # Distance moyenne entre les yeux dans le monde réel

def calculate_distance(point1, point2, image_width, image_height):
    """Calcule la distance euclidienne entre deux points."""
    x1, y1 = int(point1.x * image_width), int(point1.y * image_height)
    x2, y2 = int(point2.x * image_width), int(point2.y * image_height)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def pixels_to_cm(pixel_distance, pixel_eye_distance):
    """Convertit une distance en pixels en centimètres."""
    return (pixel_distance / pixel_eye_distance) * REAL_EYE_DISTANCE_CM

def determine_face_shape(face_width, face_height, jaw_width, forehead_width):
    """Détermine la forme du visage en fonction des proportions."""
    ratio_width_height = face_width / face_height

    if 0.7 <= ratio_width_height <= 0.8:
        return "Oval"
    elif 0.8 <= ratio_width_height <= 1.0:
        if jaw_width >= forehead_width * 0.9:
            return "Carre"
        else:
            return "Rond"
    elif 0.5 <= ratio_width_height < 0.7:
        return "Rectangulaire"
    elif face_width > jaw_width and face_width > forehead_width:
        return "Triangulaire"
    else:
        return "Indéterminée"

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

        # Dessiner les landmarks et analyser le visage
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape

                # Landmarks pour les yeux
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                # Landmarks pour la hauteur et la largeur du visage
                chin = face_landmarks.landmark[152]
                forehead = face_landmarks.landmark[10]
                left_cheek = face_landmarks.landmark[234]
                right_cheek = face_landmarks.landmark[454]

                # Largeur de la mâchoire
                jaw_left = face_landmarks.landmark[234]
                jaw_right = face_landmarks.landmark[454]

                # Calculs en pixels
                eye_distance_pixels = calculate_distance(left_eye, right_eye, w, h)
                face_height_pixels = calculate_distance(chin, forehead, w, h)
                face_width_pixels = calculate_distance(left_cheek, right_cheek, w, h)
                jaw_width_pixels = calculate_distance(jaw_left, jaw_right, w, h)
                forehead_width_pixels = calculate_distance(left_cheek, right_cheek, w, h)

                # Conversion en centimètres
                if eye_distance_pixels > 0:  # Éviter la division par zéro
                    eye_distance_cm = REAL_EYE_DISTANCE_CM
                    face_height_cm = pixels_to_cm(face_height_pixels, eye_distance_pixels)
                    face_width_cm = pixels_to_cm(face_width_pixels, eye_distance_pixels)
                    jaw_width_cm = pixels_to_cm(jaw_width_pixels, eye_distance_pixels)
                    forehead_width_cm = pixels_to_cm(forehead_width_pixels, eye_distance_pixels)
                else:
                    eye_distance_cm = face_height_cm = face_width_cm = jaw_width_cm = forehead_width_cm = 0

                # Déterminer la forme du visage
                face_shape = determine_face_shape(face_width_cm, face_height_cm, jaw_width_cm, forehead_width_cm)

                # Afficher les résultats
                cv2.putText(image, f"Forme: {face_shape}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(image, f"Distance yeux: {eye_distance_cm:.2f} cm", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(image, f"Hauteur visage: {face_height_cm:.2f} cm", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(image, f"Largeur visage: {face_width_cm:.2f} cm", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Dessiner les landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        # Afficher l'image avec les résultats
        cv2.imshow('Face Analysis (cm)', image)

        # Quitter avec la touche 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
