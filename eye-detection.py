# -*- coding: utf-8 -*-


import cv2 as cv

# CHARGEMENT DES CLASSIFICATEURS EN CASCADE PRE-ENTRAINES
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

# CHARGEMENT DES IMAGES
img = cv.imread('nom_image.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)



# EXECUTION DE LA DETECTION DE VISAGE
# detectMultiScale(image, scale factor, number of neighbors)
faces = face_cascade.detectMultiScale(gray, 1.1, 8)


# AFFICHAGE DES VISAGES
for face in faces:
    x, y, w, h = face

    # dessiner le rectangle sur l'image principale
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
# EXECUTION DETECTION DES YEUX
eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)


# AFFICAHGE DES YEUX
for (ex, ey, ew, eh) in eyes:
    # dessiner le rectangle autour des yeux sur l'image principale
    cv.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    
# affichage de l'image principale avec le rectangle
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()