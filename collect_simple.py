import os
import cv2
import time
import sys

# Create data directory if it doesn't exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_user_input():
    while True:
        try:
            print("\nWhich letter do you want to collect? (0-25)")
            print("0=A, 1=B, 2=C, ..., 25=Z")
            number = input("Enter number: ").strip()
            
            if not number.isdigit() or not (0 <= int(number) <= 25):
                print("Please enter a valid number between 0 and 25")
                continue
                
            return int(number)
        except EOFError:
            print("\nError reading input. Please try again.")
            continue
        except KeyboardInterrupt:
            print("\nExiting program.")
            sys.exit(0)

def collect_images():
    print("Starting webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Wait a moment for the camera to initialize
    time.sleep(1)
    
    try:
        # Get the letter number with error handling
        number = get_user_input()
        letter = chr(65 + number)  # Convert to letter (A=65 in ASCII)
        
        # Create directory for this letter
        letter_dir = os.path.join(DATA_DIR, str(number))
        if not os.path.exists(letter_dir):
            os.makedirs(letter_dir)
            
        print(f"\nCollecting images for letter {letter}")
        print("Position your hand and press 'q' when ready")
        
        # Show webcam feed until user is ready
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                return
                
            # Show the frame
            cv2.putText(frame, f"Ready for letter {letter}? Press 'q'", 
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
            
            # Wait for 'q' press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC key
                print("\nCancelled by user")
                return
        
        # Collect 100 images
        print("\nCollecting images... Keep your hand steady!")
        for i in range(100):
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                return
            
            # Save the image
            image_path = os.path.join(letter_dir, f"{i}.jpg")
            cv2.imwrite(image_path, frame)
            
            # Show progress
            cv2.putText(frame, f"Collecting {letter}: {i+1}/100", 
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
            
            # Check for ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                print("\nCollection interrupted by user")
                return
            
        print(f"\nDone! Collected 100 images for letter {letter}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_images()
