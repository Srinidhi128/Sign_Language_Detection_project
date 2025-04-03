import os
import cv2

# Map numbers to letters for easier reference
LETTER_MAP = {str(i): chr(65 + i) for i in range(26)}  # 0->A, 1->B, etc.

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 100

def main():
    cap = cv2.VideoCapture(0)  # Back to camera index 0
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Get the class number from user
    print("\nEnter the number for the letter you want to collect:")
    print("\n".join([f"{num}: {letter}" for num, letter in LETTER_MAP.items()]))
    
    try:
        j = input("\nEnter number (0-25): ").strip()
        
        if not j.isdigit() or int(j) not in range(26):
            print("Invalid input. Please enter a number between 0-25.")
            cap.release()
            cv2.destroyAllWindows()
            return
        
        j = int(j)
        folder_path = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        print(f'Collecting data for letter {LETTER_MAP[str(j)]} (class {j})')
        print("Press 'Q' when ready to start collecting images")
        
        # Wait for Q press
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            cv2.putText(frame, f'Ready to collect letter {LETTER_MAP[str(j)]}? Press "Q"!', 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Collect Images', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 27:  # ESC key
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Collect images
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            cv2.putText(frame, f'Collecting {LETTER_MAP[str(j)]}: {counter}/{dataset_size}', 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Collect Images', frame)
            
            # Save the image
            cv2.imwrite(os.path.join(folder_path, f'{counter}.jpg'), frame)
            counter += 1
            
            # Small delay to not overwhelm the system
            cv2.waitKey(25)
        
        print(f"\nFinished collecting {dataset_size} images for letter {LETTER_MAP[str(j)]}")
        
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
