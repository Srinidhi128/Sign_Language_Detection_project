import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './datawords'

data = []
labels = []

try:
    # Process each directory (class)
    for dir_ in tqdm(os.listdir(DATA_DIR), desc="Processing classes"):
        dir_path = os.path.join(DATA_DIR, dir_)
        if os.path.isdir(dir_path):
            # Process each image in the directory
            for img_path in tqdm(os.listdir(dir_path), desc=f"Processing {dir_}"):
                img_path_full = os.path.join(dir_path, img_path)
                if os.path.isfile(img_path_full):
                    try:
                        data_aux = []
                        x_ = []
                        y_ = []

                        img = cv2.imread(img_path_full)
                        if img is None:
                            print(f"Failed to load image: {img_path_full}")
                            continue

                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        H, W, _ = img_rgb.shape

                        results = hands.process(img_rgb)
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y
                                    x_.append(x)
                                    y_.append(y)

                                for i in range(len(hand_landmarks.landmark)):
                                    x = hand_landmarks.landmark[i].x
                                    y = hand_landmarks.landmark[i].y
                                    data_aux.append(x - min(x_))
                                    data_aux.append(y - min(y_))

                            # If only one hand is detected, pad the data
                            if len(results.multi_hand_landmarks) == 1:
                                while len(data_aux) < 84:
                                    data_aux.append(0)

                            if len(results.multi_hand_landmarks) == 2:
                                while len(data_aux) < 84:  # Ensure 84 features (42 for each hand)
                                    data_aux.append(0)

                            data.append(data_aux)
                            labels.append(int(dir_))

                    except Exception as e:
                        print(f"Error processing {img_path_full}: {str(e)}")
                        continue

    # Save the processed data
    print("Saving processed data...")
    pickle_path = os.path.join(os.path.dirname(__file__), 'datawords.pickle')
    f = open(pickle_path, 'wb')
    pickle.dump({'datawords': data, 'labels': labels}, f)
    f.close()
    print("Dataset creation completed successfully!")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving partial data...")
    if data and labels:
        pickle_path = os.path.join(os.path.dirname(__file__), 'datawords.pickle')
        f = open(pickle_path, 'wb')
        pickle.dump({'datawords': data, 'labels': labels}, f)
        f.close()
        print("Partial data saved successfully!")
    else:
        print("No data to save.")

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    cv2.destroyAllWindows()
