import cv2

# Open and connect to the default camera

capture = cv2.VideoCapture(0)

while True :
    
    # Check the availability of the camera 
    ret , frame = capture.read()
    
    # if the camera is available ret would be true 
    
    if ret :
        cv2.imshow('Camera live feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    else :
        print ('Failed to find Camera')
        break
    


capture.release()
cv2.destroyAllWindows()