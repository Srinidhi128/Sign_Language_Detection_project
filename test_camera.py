import cv2

# Open the webcam
cap = cv2.VideoCapture(1)  # Changed from 0 to 1

if not cap.isOpened():
    print("Error: Could not open webcam")
else:
    print("Webcam opened successfully")
    print("Press 'q' to quit")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Display the frame
        cv2.imshow('Webcam Test', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
print("Test complete")
