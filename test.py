import cv2
import os
import random
import string

save_dir = os.getcwd()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
    
# Function to generate a random filename
def generate_filename():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".jpg"


# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Face Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if key == ord('c'):
        # Generate a random filename
        filename = generate_filename()
        
        # Construct the path where the image will be saved
        filepath = os.path.join(f"{save_dir}/face-id/", filename)
        
        # Save the frame as an image file
        cv2.imwrite(filepath, frame)
        print(f"Image saved: {filepath}")
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
