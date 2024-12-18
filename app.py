import cv2
import dlib
import numpy as np

# Lade den Gesichtserkennungsmodell
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def face_swap(image1, image2):
    # Konvertiere beide Bilder in Graustufen
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detektiere Gesichter in beiden Bildern
    faces1 = detector(gray1)
    faces2 = detector(gray2)

    if len(faces1) == 0 or len(faces2) == 0:
        print("Kein Gesicht gefunden!")
        return None

    # Nimm das erste Gesicht aus beiden Bildern (falls mehrere Gesichter erkannt werden)
    face1 = faces1[0]
    face2 = faces2[0]

    # Erhalte Gesichtslandmarken
    landmarks1 = predictor(gray1, face1)
    landmarks2 = predictor(gray2, face2)

    # Extrahiere die Gesichtsregionen
    points1 = np.array([(p.x, p.y) for p in landmarks1.parts()])
    points2 = np.array([(p.x, p.y) for p in landmarks2.parts()])

    # Berechne die Masken für die Gesichter
    mask1 = np.zeros_like(gray1)
    mask2 = np.zeros_like(gray2)
    
    # Zeichne die Gesichter auf die Masken
    cv2.fillConvexPoly(mask1, points1, 255)
    cv2.fillConvexPoly(mask2, points2, 255)

    # Hole die Gesichter selbst mit einer Maske
    face1_img = cv2.bitwise_and(image1, image1, mask=mask1)
    face2_img = cv2.bitwise_and(image2, image2, mask=mask2)

    # Hier könntest du den Face-Swap-Algorithmus einfügen, um die Gesichter zu tauschen
    # (dies könnte z.B. durch Warping und Farbangleichung erfolgen)

    # Zeige das Ergebnis (für Testzwecke)
    cv2.imshow("Face 1", face1_img)
    cv2.imshow("Face 2", face2_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Lade die beiden Bilder
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

face_swap(image1, image2)
