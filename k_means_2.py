# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 20:34:04 2020

@author: steven
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from tensorflow.python.platform import gfile
import random
import re


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
    help="optional path to input image to segment")
ap.add_argument("-o", "--output",
    help="optional path to output processed images")
args = vars(ap.parse_args())


string = args["image"].split("\\")
print(string)
string = re.sub('[^0-9]', '', string[len(string)-1])
c = string
        



# leer imagen
image_org = cv2.imread(args["image"])

# convertirla a rgb
image = cv2.cvtColor(image_org, cv2.COLOR_BGR2RGB)

print(image.shape)


# cambiar la forma a un arreglo de 2 dimensiones de pixeles y 3 bandas de color (RGB)
pixel_values = image.reshape((-1, 3))
# Convertir pixeles a flotantes
pixel_values = np.float32(pixel_values)

# definir criterio de parada
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# Definir numero de clusters (K) 
k = 4 #Se usaron 4 más que todo por la presencia de nubes, si se usaban las nubes se segmentaban como parametros de mineria

#Implementación de K-Means en OpenCV
compactness, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir nuevamennte los pixeles a valores de 8 bits
centers = np.uint8(centers)

# Estirar arreglo de etiquetas
labels = labels.flatten()

# Asigna cada pixel al valor de sus labels 
segmented_image = centers[labels]

# Redimensiona nuevamente imagen al tamaño de la original
segmented_image = segmented_image.reshape(image.shape)

# Muestra la imagen segmentada
cv2.imshow(segmented_image)


# Se saca una copia de la imagen segmentada
masked_image = np.copy(image)
# Se convierte nuevamente a un arreglo de 2 dimensiones de pixeles
masked_image = masked_image.reshape((-1, 3))
# Identificacion del Cluster

cluster_2 = 2
cluster_3 = 3
cluster_1 = 1
cluster_0 = 0

#Valores de Color de los patrones segmentados (la idea es dejar en blanco los clusters de interes)
#Estos cambios deben hacerse de forma manual, es una anotación semisupervisada. 
masked_image[labels == cluster_2] = [255, 255, 255]
masked_image[labels == cluster_3] = [0, 0, 0]
masked_image[labels == cluster_1] = [0, 0, 0]
masked_image[labels == cluster_0] = [255, 255, 255]

# Convertir nuevamente a la forma de la imagen original
masked_image = masked_image.reshape(image.shape)

#Convertir imagen a escala de grises
masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

#Hacer umbralizacion (Thresholding) la imagen para destacar las regiones que son de interes
(T, thresh) = cv2.threshold(masked_image, 160, 255, cv2.THRESH_BINARY)

#Muestra imagen umbralizada
cv2.imshow("Threshold Binary", thresh)
#masked_image = cv2.bitwise_not(masked_image)
# Mostrar mascara - ROI
cv2.imshow(masked_image)


#Guardar imagen en el disco
cv2.imwrite(args["output"]+"/mask_k_means_" + str(c)+ ".jpg", masked_image)

#Operacion AND entre la imagen original y la mascara
masked = cv2.bitwise_and(image_org, image_org, mask = thresh)

#Mostrar imagen enmascarada
cv2.imshow("Mask Applied to Image", masked)

#guardar imagen en disco
cv2.imwrite(args["output"]+"/mask_k_means_result_" + str(c)+ ".jpg", masked)
cv2.waitKey(0)

print(str(c))