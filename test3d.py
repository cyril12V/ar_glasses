import os
import cv2
import mediapipe as mp
import numpy as np
import pyrender
import trimesh

# Supprimer les avertissements TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialisation Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Charger le modèle 3D des lunettes
glasses_file = "lunette3d1.obj"
glasses_scene = trimesh.load(glasses_file)

# Si le fichier est une scène, extraire les meshes individuels
if isinstance(glasses_scene, trimesh.Scene):
    glasses_mesh = trimesh.util.concatenate(glasses_scene.dump())
else:
    glasses_mesh = glasses_scene

# Créer une scène PyRender
scene = pyrender.Scene()

# Ajouter les lunettes à la scène
glasses_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(glasses_mesh))
scene.add_node(glasses_node)

# Initialiser la caméra virtuelle pour PyRender
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_node = scene.add(camera, pose=np.eye(4))

# Créer le renderer
renderer = pyrender.OffscreenRenderer(640, 480)

# Initialisation de la caméra OpenCV
cap = cv2.VideoCapture(0)

# Utiliser Mediapipe pour détecter les landmarks
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Impossible de lire la caméra.")
            break

        # Convertir l'image en RGB pour Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détecter les landmarks
        results = face_mesh.process(frame_rgb)

        # Revenir à BGR pour affichage OpenCV
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Si des landmarks sont détectés
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Récupérer les points des yeux
                h, w, _ = frame.shape
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                # Calculer les coordonnées des yeux en pixels
                left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
                right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

                # Calculer le centre entre les yeux et la distance
                center_x = (left_eye_x + right_eye_x) / 2
                center_y = (left_eye_y + right_eye_y) / 2
                distance = np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2)

                # Calculer la transformation pour positionner les lunettes
                scale = distance / 100  # Ajuster l'échelle en fonction de la distance entre les yeux
                translation = np.array([[1, 0, 0, center_x / w - 0.5],
                                         [0, 1, 0, -(center_y / h - 0.5)],
                                         [0, 0, 1, -1],  # Positionner devant la caméra
                                         [0, 0, 0, 1]])
                scaling = np.diag([scale, scale, scale, 1])

                # Appliquer la transformation
                transformation_matrix = translation @ scaling
                scene.set_pose(glasses_node, pose=transformation_matrix)

        # Rendu avec PyRender
        rendered_color, _ = renderer.render(scene)

        # Afficher le rendu avec OpenCV
        cv2.imshow("Lunettes 3D", rendered_color)

        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
renderer.delete()
