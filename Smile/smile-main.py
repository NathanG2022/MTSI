import cv2

# Load the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_smile.xml')

# Start the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read the frame from the webcam
    successful_frame_read, frame = webcam.read()
    
    if not successful_frame_read:
        break
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame)
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)
        
        # Get the region of interest (ROI) for the face
        face_roi = gray_frame[y:y+h, x:x+w]
        
        # Detect smiles within the ROI
        smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=20)
        
        # If any smiles are detected, put text on the frame
        if len(smiles) == 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))
    
    # Display the frame with face and smile detection
    cv2.imshow('Real-time Face and Smile Detection', frame)
    
    # Break the loop if 'Q' or 'q' is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):                
        break
# Release the webcam and destroy all windows
webcam.release()
cv2.destroyAllWindows()
