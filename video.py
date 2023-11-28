import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load the video
input_video_path = input("Enter Image Path: ")
cap = cv2.VideoCapture(input_video_path)

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = 'output_video_blurred.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30) 
        frame[y:y+h, x:x+w] = face
    out.write(frame)
cap.release()
out.release()
cv2.destroyAllWindows()