import os
import cv2
import mediapipe as mp

# Supprimer les avertissements TensorFlow et Mediapipe
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation de Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

        # Dessiner les landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # Afficher l'image avec les landmarks
        cv2.imshow('Face Mesh', image)

        # Quitter avec la touche 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
