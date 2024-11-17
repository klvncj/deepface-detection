import cv2
import os
import random
import string
from deepface import DeepFace



## To add a drawing on the code to make some face analysis thing 


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

#clean the output 
def clean_data(data):
    cleaned_data = []
    for person in data:
        # Extracting dominant gender, emotion, and race with their percentages
        dominant_gender = f"{person['dominant_gender'].capitalize()} ({round(person['gender'][person['dominant_gender']] , 2)}%)"
        dominant_emotion = f"{person['dominant_emotion'].capitalize()} ({round(person['emotion'][person['dominant_emotion']] , 2)}%)"
        dominant_race = f"{person['dominant_race'].capitalize()} ({round(person['race'][person['dominant_race']] , 2)}%)"
        
        # Cleaning the region data
        region_data = {k: person['region'][k] for k in ['x', 'y', 'w', 'h']}

        # Creating a cleaned structure
        cleaned_person = {
            "age": person['age'],
            "region": region_data,
            "face_confidence": person['face_confidence'],
            "dominant_gender": dominant_gender,
            "dominant_emotion": dominant_emotion,
            "dominant_race": dominant_race
        }
        cleaned_data.append(cleaned_person)
    return cleaned_data

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
