import cv2
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread('Plaque.jpg')
if image is None:
    print("Impossible de charger l'image.")
    exit()

# Convertir en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Seuil adaptatif
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Détection de bords Canny
edges = cv2.Canny(adaptive_thresh, 100, 200)

# Érosion et dilatation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

# Détection des contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"Nombre de contours détectés : {len(contours)}")

# Dessiner les contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w < 50 and h < 50:  # Ajustez ces valeurs selon la taille des caractères
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Afficher les images
plt.figure(figsize=(12, 12))
plt.subplot(221), plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), plt.title('Niveaux de Gris')
plt.subplot(222), plt.imshow(cv2.cvtColor(adaptive_thresh, cv2.COLOR_BGR2RGB)), plt.title('Seuil Adaptatif')
plt.subplot(223), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)), plt.title('Canny Edge Detection')
plt.subplot(224), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Résultat Final')
plt.show()
