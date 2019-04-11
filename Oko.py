import cv2
from matplotlib import pyplot as plt

def wczytajObraz(nazwaPliku):
    obraz = cv2.imread(nazwaPliku)
    return obraz

def zastosujMaske(obraz, nazwaPliku):
    obrazMaska = cv2.imread(nazwaPliku)
    obrazMaska = cv2.cvtColor(obrazMaska, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(obrazMaska, 127, 255, cv2.THRESH_BINARY)

    obrazZMaska = cv2.bitwise_and(obraz, obraz, mask = mask)
    return obrazZMaska

img = wczytajObraz("images/01_dr.JPG")
plt.imshow(img)
plt.show()

plt.imshow(zastosujMaske(img, "mask/01_dr_mask.tif"))
plt.show()