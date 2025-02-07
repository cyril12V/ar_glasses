# OptiMorph - Essayez vos lunettes en AR

## 📌 Description
OptiMorph est une application web permettant d'essayer des lunettes en réalité augmentée (AR) directement depuis votre navigateur. Grâce à la reconnaissance faciale de Mediapipe et un serveur Flask avec WebSocket, l'application analyse votre visage et superpose dynamiquement des modèles de lunettes.

## 🚀 Fonctionnalités
- Détection et analyse du visage avec Mediapipe.
- Calcul des dimensions du visage et reconnaissance de la forme.
- Superposition en temps réel des lunettes via OpenCV.
- Interface utilisateur interactive avec Flask et TailwindCSS.
- WebSocket pour un streaming vidéo fluide.

## 🛠️ Installation

### 1️⃣ Cloner le projet
```sh
git clone https://github.com/cyril12V/ar_glasses.git
cd ar_glasses
```

### 2️⃣ Installer les dépendances
Assurez-vous d'avoir Python 3 installé.
```sh
pip install -r requirements.txt
```

### 3️⃣ Lancer l'application
```sh
python app.py
```
Puis ouvrez votre navigateur et accédez à :
```
http://127.0.0.1:5000
```

## 📷 Détection du visage et essayage AR
L'application capture le flux vidéo, détecte le visage et superpose les lunettes en fonction de la distance entre les yeux.

### 🔍 Algorithme principal
1. Capture du flux vidéo via OpenCV.
2. Détection des landmarks faciaux avec Mediapipe.
3. Calcul de la taille et de la forme du visage.
4. Superposition des lunettes avec redimensionnement et rotation.
5. Envoi des images traitées via WebSocket.

## 📁 Structure du projet
```
📦 ar_glasses
├── 📂 static          # Fichiers statiques (images, CSS, JS)
├── 📂 templates       # Pages HTML
├── app.py            # Serveur Flask & traitement AR
├── requirements.txt   # Dépendances Python
├── README.md         # Documentation du projet
```

## 🌐 Routes principales
- `/` : Page d'accueil
- `/index` : Interface d'essayage AR
- `/catalogue` : Liste des lunettes disponibles
- `/analyse-visage` : Page d'analyse du visage
- `/api/analyse-visage` : API d'analyse du visage

## ⚡ Technologies utilisées
- **Python** (Flask, OpenCV, Mediapipe, NumPy)
- **JavaScript** (Socket.IO, TailwindCSS)
- **HTML/CSS** (Templates Jinja2, TailwindCSS)

## 🖥️ Développement & Collaboration
Si vous souhaitez contribuer, clonez le repo, créez une branche et soumettez une PR !
```sh
git checkout -b feature-nouvelle-fonction
```

## 🔗 Liens utiles
- Mediapipe : https://developers.google.com/mediapipe
- Flask : https://flask.palletsprojects.com/
- OpenCV : https://opencv.org/

👓 **Profitez de l'essayage en AR avec OptiMorph !** 🚀

