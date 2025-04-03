import cv2
import mediapipe as mp
import pickle
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter as ctk

class LiveDetector:
    def __init__(self):
        # Create the main window
        self.window = ctk.CTk()
        self.window.title("Sign Language Live Detection")
        self.window.geometry("800x600")
        
        # Initialize mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load the pre-trained model
        try:
            with open('model.p', 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
            print("Model loaded successfully!")
        except:
            print("Error: Could not load model.p")
            self.model = None

        # Create GUI elements
        self.setup_gui()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            
        # Start video stream
        self.update_frame()

    def setup_gui(self):
        # Main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video label (where the camera feed will be displayed)
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(expand=True)
        
        # Detected letter label
        self.letter_label = ctk.CTkLabel(
            self.main_frame, 
            text="Detected Letter: -",
            font=("Arial", 24, "bold")
        )
        self.letter_label.pack(pady=10)

    def update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
            
        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with mediapipe
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks and make prediction if hand is detected
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks on frame
            self.mp_draw.draw_landmarks(
                frame_rgb, 
                landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Make prediction if model is loaded
            if self.model is not None:
                # Prepare data for prediction
                data_point = []
                for lm in landmarks.landmark:
                    data_point.extend([lm.x, lm.y, lm.z])
                
                # Make prediction
                prediction = self.model.predict([data_point])
                predicted_letter = chr(65 + prediction[0])  # Convert to letter (A=65 in ASCII)
                
                # Update letter label
                self.letter_label.configure(text=f"Detected Letter: {predicted_letter}")
                
                # Draw prediction on frame
                cv2.putText(
                    frame_rgb, 
                    predicted_letter, 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    2, 
                    (0, 255, 0), 
                    2
                )
        else:
            self.letter_label.configure(text="Detected Letter: -")
        
        # Convert frame to PhotoImage and display
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        
        # Schedule the next update
        self.window.after(10, self.update_frame)

    def run(self):
        self.window.mainloop()
        
    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveDetector()
    try:
        app.run()
    finally:
        app.cleanup()
