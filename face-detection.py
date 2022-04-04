# -*- coding: utf-8 -*-


import cv2 as cv

# CHARGEMENT DES CLASSIFICATEURS EN CASCADE PRE-ENTRAINES
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


# CHARGEMENT DES IMAGES
img = cv.imread('nom_image.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



# EXECUTION DE LA DETECTION DE VISAGE
# detectMultiScale(image, scale factor, number of neighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)



# AFFICHAGE DES VISAGES
i = 0
for face in faces:
    x, y, w, h = face

    # dessiner le rectangle sur l'image principale
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Paramétres : 
    # img => l'image
    # (x, y) => coordonnées du coin en haut à gauche de la détection
    # (x + w, y + h) => coordonnées du coin en bas à droite sur le visage
    # >>> (abscisse + largeur , ordonnée + hauteur)
    # (0, 255, 0) couleur du rectangle
    # (0, 255, 0), 2) => 2 est l'épaisseur de la ligne du rectangle
    
    # Extraction des visages de l'image principale
    # OpenCV et Numpy : y <-> row et x <-> col
    face = img[y:y+h, x:x+w]
    
    # afficher face0, face1, face2, etc ...
    cv.imshow('face{}'.format(i), face)
    i += 1
    
# affichage de l'image principale avec le rectangle
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()