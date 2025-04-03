import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./modelwords.p', 'rb'))
model = model_dict['modelwords']

# Open the camera
cap = cv2.VideoCapture(0)

# Setup mediapipe for hand landmark detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for prediction
labels_dict = {
    0: 'Hello', 1: "Bye", 2: 'man', 3: 'Women', 4: 'Thankyou', 
    5: 'welcome', 6: 'Sorry', 7: 'Namasthe', 8: 'Yes', 9: 'No',
    10: 'Good', 11: 'Bad', 12: 'Correct', 13: 'Wrong', 14: 'Easy', 
    15: 'Difficult'
}

while True:
    data_aux = []  # List to hold the features
    x_ = []  # List to hold the x coordinates
    y_ = []  # List to hold the y coordinates

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    H, W, _ = frame.shape

    # Convert to RGB for processing by mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        # Process each hand detected
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize and append the features for the hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))  # Normalize x values
                data_aux.append(y - min(y_))  # Normalize y values

        # If only one hand is detected, pad the data to 84 features (42 for each hand)
        if len(results.multi_hand_landmarks) == 1:
            while len(data_aux) < 84:  # Pad to 84 features (42 for one hand)
                data_aux.append(0)

        # If two hands are detected, ensure 84 features (42 for each hand)
        if len(results.multi_hand_landmarks) == 2:
            while len(data_aux) < 84:  # Ensure 84 features (42 for each hand)
                data_aux.append(0)

        # Prediction based on extracted features
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and prediction text
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the result
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
