import cv2
import numpy as np

# Charger l'image de la plaque d'immatriculation
image_path = "Plaque.jpg"
image = cv2.imread(image_path)
if image is None:
    print("Erreur lors du chargement de l'image")
    exit()

# Afficher l'image originale
cv2.imshow("Image Originale", image)
cv2.waitKey(0)

# Convertir l'image en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Afficher l'image en niveaux de gris
cv2.imshow("Niveaux de Gris", gray)
cv2.waitKey(0)

# Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Afficher l'image floutée
cv2.imshow("Image Floutee", blurred)
cv2.waitKey(0)

# Utiliser le seuil Otsu pour la binarisation
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Afficher l'image binaire
cv2.imshow("Image Binaire", binary)
cv2.waitKey(0)

# Utiliser la détection de contours
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dessiner des rectangles autour de chaque contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Dessiner le rectangle si les dimensions sont dans les limites spécifiées
    if 20 < w < 100 and 40 < h < 200:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Afficher l'image avec les rectangles autour des lettres détectées
cv2.imshow("Plaque avec segmentation des lettres", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
