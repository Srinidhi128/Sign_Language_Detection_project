import tkinter as tk
from tkinter import ttk
import cv2
import os
import pickle
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import customtkinter as ctk
from PIL import Image, ImageTk

class SignLanguageApp:
    def __init__(self):
        self.app = ctk.CTk()
        self.app.title("Sign Language Detector")
        self.app.geometry("1200x800")
        
        # Initialize mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize variables
        self.current_letter = 0
        self.counter = 0
        self.dataset_size = 100
        self.is_collecting = False
        self.is_detecting = False
        self.model = None
        self.cap = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frames
        self.control_frame = ctk.CTkFrame(self.app)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        self.video_frame = ctk.CTkFrame(self.app)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control Panel
        ctk.CTkLabel(self.control_frame, text="Sign Language Detector", font=("Arial", 20, "bold")).pack(pady=10)
        
        # Collection Controls
        collection_frame = ctk.CTkFrame(self.control_frame)
        collection_frame.pack(pady=10, padx=5, fill=tk.X)
        
        ctk.CTkLabel(collection_frame, text="Data Collection", font=("Arial", 16, "bold")).pack(pady=5)
        
        # Letter selection
        letter_frame = ctk.CTkFrame(collection_frame)
        letter_frame.pack(pady=5, fill=tk.X)
        
        self.letter_var = tk.StringVar(value='A')
        letters = [chr(65 + i) for i in range(26)]  # A to Z
        
        ctk.CTkLabel(letter_frame, text="Select Letter:").pack(side=tk.LEFT, padx=5)
        letter_menu = ctk.CTkOptionMenu(letter_frame, values=letters, variable=self.letter_var)
        letter_menu.pack(side=tk.LEFT, padx=5)
        
        # Collection buttons
        btn_frame = ctk.CTkFrame(collection_frame)
        btn_frame.pack(pady=5, fill=tk.X)
        
        self.collect_btn = ctk.CTkButton(btn_frame, text="Start Collection", command=self.toggle_collection)
        self.collect_btn.pack(side=tk.LEFT, padx=5)
        
        # Training Controls
        training_frame = ctk.CTkFrame(self.control_frame)
        training_frame.pack(pady=10, padx=5, fill=tk.X)
        
        ctk.CTkLabel(training_frame, text="Model Training", font=("Arial", 16, "bold")).pack(pady=5)
        
        self.train_btn = ctk.CTkButton(training_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(pady=5)
        
        # Detection Controls
        detection_frame = ctk.CTkFrame(self.control_frame)
        detection_frame.pack(pady=10, padx=5, fill=tk.X)
        
        ctk.CTkLabel(detection_frame, text="Live Detection", font=("Arial", 16, "bold")).pack(pady=5)
        
        self.detect_btn = ctk.CTkButton(detection_frame, text="Start Detection", command=self.toggle_detection)
        self.detect_btn.pack(pady=5)
        
        # Status
        self.status_label = ctk.CTkLabel(self.control_frame, text="Ready", font=("Arial", 12))
        self.status_label.pack(pady=10)
        
        # Video display
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True)
        
    def update_status(self, message):
        self.status_label.configure(text=message)
        
    def toggle_collection(self):
        if not self.is_collecting:
            self.start_collection()
        else:
            self.stop_collection()
            
    def start_collection(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_status("Error: Could not open webcam")
                return
                
        self.is_collecting = True
        self.collect_btn.configure(text="Stop Collection")
        self.current_letter = ord(self.letter_var.get()) - 65
        self.counter = 0
        
        # Create directory if it doesn't exist
        folder_path = os.path.join('./data', str(self.current_letter))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.update_frame()
        
    def stop_collection(self):
        self.is_collecting = False
        self.collect_btn.configure(text="Start Collection")
        if self.cap is not None and not self.is_detecting:
            self.cap.release()
            self.cap = None
            
    def train_model(self):
        self.update_status("Training model...")
        try:
            data_dict = {
                'data': [],
                'labels': []
            }
            
            # Load the data
            for i in range(26):  # A to Z
                folder_path = os.path.join('./data', str(i))
                if os.path.exists(folder_path):
                    for img_name in os.listdir(folder_path):
                        img_path = os.path.join(folder_path, img_name)
                        img = cv2.imread(img_path)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = self.hands.process(img_rgb)
                        
                        if results.multi_hand_landmarks:
                            landmarks = results.multi_hand_landmarks[0]
                            data_point = []
                            for lm in landmarks.landmark:
                                data_point.extend([lm.x, lm.y, lm.z])
                            data_dict['data'].append(data_point)
                            data_dict['labels'].append(i)
            
            # Train the model
            X = np.array(data_dict['data'])
            y = np.array(data_dict['labels'])
            
            self.model = RandomForestClassifier()
            self.model.fit(X, y)
            
            # Save the model
            with open('model.p', 'wb') as f:
                pickle.dump({'model': self.model}, f)
                
            self.update_status("Model trained successfully!")
            
        except Exception as e:
            self.update_status(f"Training error: {str(e)}")
            
    def toggle_detection(self):
        if not self.is_detecting:
            self.start_detection()
        else:
            self.stop_detection()
            
    def start_detection(self):
        if self.model is None:
            try:
                with open('model.p', 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
            except:
                self.update_status("Error: No trained model found")
                return
                
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.update_status("Error: Could not open webcam")
                return
                
        self.is_detecting = True
        self.detect_btn.configure(text="Stop Detection")
        self.update_frame()
        
    def stop_detection(self):
        self.is_detecting = False
        self.detect_btn.configure(text="Start Detection")
        if self.cap is not None and not self.is_collecting:
            self.cap.release()
            self.cap = None
            
    def update_frame(self):
        if self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.update_status("Error: Could not read frame")
            return
            
        frame = cv2.flip(frame, 1)  # Mirror the image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if self.is_collecting:
                # Save image for training
                img_path = os.path.join('./data', str(self.current_letter), f'{self.counter}.jpg')
                cv2.imwrite(img_path, frame)
                self.counter += 1
                
                if self.counter >= self.dataset_size:
                    self.stop_collection()
                    self.update_status(f"Finished collecting images for {self.letter_var.get()}")
                else:
                    self.update_status(f"Collecting {self.letter_var.get()}: {self.counter}/{self.dataset_size}")
                    
            elif self.is_detecting and self.model is not None:
                # Prepare data for prediction
                data_point = []
                for lm in landmarks.landmark:
                    data_point.extend([lm.x, lm.y, lm.z])
                    
                # Make prediction
                prediction = self.model.predict([data_point])
                predicted_letter = chr(65 + prediction[0])
                
                # Display prediction
                cv2.putText(frame, predicted_letter, (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                
        # Convert frame to PhotoImage and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=img)
        self.video_label.image = img
        
        if self.is_collecting or self.is_detecting:
            self.video_label.after(10, self.update_frame)
            
    def run(self):
        self.app.mainloop()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
