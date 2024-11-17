import cv2
import os
import random
import string
from deepface import DeepFace

# Set the directory to save captured images
save_dir = os.getcwd()
capture_dir = os.path.join(save_dir, "face-id", "faces")
if not os.path.exists(capture_dir):
    os.makedirs(capture_dir)

def newFilename():
    allImage = os.listdir(capture_dir)
    value = len(allImage)
    return f"img-0x000001F9DF3BA2A0-{value + 1}.jpg"
    



# Function to generate a random filename
def generate_filename():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10)) + ".jpg"


# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

print("Press 'k' to capture and analyze a face. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break

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
    if key == ord('q'):
        print("Exiting the application.")
        break
    
    if key == ord('k'):
        # Generate a random filename
        filename = newFilename()
        
        # Construct the path where the image will be saved
        filepath = os.path.join(capture_dir, filename)
        
        # Save the frame as an image file
        cv2.imwrite(filepath, frame)
        print(f"Image saved: {filepath}")
        
        try:
            # Perform face analysis using DeepFace
            analysis = DeepFace.analyze(img_path=filepath, actions=['age', 'gender', 'emotion', 'race'])
            
            # Print the analysis results
            print("Face Analysis Results:")
            print(analysis)
        
        except Exception as e:
            print(f"Error during face analysis: {str(e)}")

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
