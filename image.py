import cv2 , colorama , numpy , os
import numpy as np
from colorama import Fore, Style
from PIL import Image
## Input Image
img_path = input(Fore.YELLOW + "Enter image path: ")
if not img_path:
  print(Fore.RED + "Invalid path")
  exit()
## Image Pretrain Data 
face_cascade_default = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# Load Image
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces_default = face_cascade_default.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
faces_alt = face_cascade_alt.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
## coded by kerncraze
faces = np.concatenate((faces_default, faces_alt), axis=0)

for (x, y, w, h) in faces:
    roi = image[y:y + h, x:x + w]
    roi = cv2.GaussianBlur(roi, (55, 55), 100)
    image[y:y + h, x:x + w] = roi
cv2.imwrite("output.jpeg", image)
## Clear Image Exif
image = Image.open('output.jpeg')
data = list(image.getdata())
image_without_exif = Image.new(image.mode, image.size)
image_without_exif.putdata(data)
os.remove('output.jpeg')
image_without_exif.save('outfile.jpeg')
image_without_exif.close()
## Process is Done 
print(f"{Fore.GREEN}Done ✅{Style.RESET_ALL}")