import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import mediapipe as mp

# Load the dataset
train_data_path = 'C:/Users/anas_/Downloads/sign_language_dataset/sign_mnist_train.csv'
test_data_path = 'C:/Users/anas_/Downloads/sign_language_dataset/sign_mnist_test.csv'

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)


X_train = train_data.iloc[:, 1:].values
Y_train = train_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values
Y_test = test_data.iloc[:, 0].values


X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0


Y_train = to_categorical(Y_train, num_classes=26)
Y_test = to_categorical(Y_test, num_classes=26)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))


model.save('sign_language_model.h5')


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


def sign_language_interpreter():
    cap = cv2.VideoCapture(0)
    labels = [chr(i) for i in range(65, 91)]  # A-Z

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append([lm.x, lm.y])

                
                hand_array = np.zeros((28, 28))
                for point in landmark_list:
                    x, y = int(point[0] * 28), int(point[1] * 28)
                    if 0 <= x < 28 and 0 <= y < 28:
                        hand_array[y, x] = 1

                
                reshaped = hand_array.reshape(1, 28, 28, 1)

                
                prediction = model.predict(reshaped)
                label = labels[np.argmax(prediction)]

            
                cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Sign Language Interpreter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sign_language_interpreter()
