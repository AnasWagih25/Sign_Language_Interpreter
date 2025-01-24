# Sign Language Interpreter Using CNN and MediaPipe

This project uses a Convolutional Neural Network (CNN) to recognize American Sign Language (ASL) gestures through a webcam, utilizing the `TensorFlow`, `OpenCV`, and `MediaPipe` libraries.

## Requirements

- Python 3.6+
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib

You can install the required dependencies using:  
`pip install tensorflow opencv-python mediapipe numpy pandas matplotlib`

## Dataset

The dataset used in this project is the **Sign Language MNIST** dataset, which contains images of ASL hand gestures for the English alphabet. You can download the dataset from [here](https://www.kaggle.com/datasets/databreh/sign-language-mnist).

### Directory Structure

project/  
│  
├── sign_language_interpreter.py  # The main script for recognizing sign language  
├── sign_language_model.h5        # The trained model  
├── sign_mnist_train.csv          # Training dataset (CSV)  
├── sign_mnist_test.csv           # Test dataset (CSV)  
└── README.md                     # Project documentation  

## How It Works

1. **Model Training**:  
   The model is trained using the **Sign Language MNIST** dataset. The images are preprocessed and reshaped to fit the model input. The model is a CNN with two convolutional layers, pooling layers, and dense layers to classify the images.

2. **Sign Language Interpreter**:  
   The webcam captures hand gestures in real-time. MediaPipe processes the video stream to detect hand landmarks. The landmarks are then converted into a 28x28 image, which is passed to the trained model for classification. The recognized sign is displayed on the screen in real-time.

3. **Run the Interpreter**:  
   To start the interpreter, simply run the following command in your terminal:  
   `python sign_language_interpreter.py`

4. **Exit the Interpreter**:  
   Press the 'q' key to exit the interpreter.

## Model Architecture

- **Input**: 28x28 grayscale images (representing hand gestures)  
- **Convolutional Layers**:  
  - Conv2D with 32 filters (3x3)  
  - Conv2D with 64 filters (3x3)  
- **Pooling**: MaxPooling2D with (2x2) kernel  
- **Fully Connected Layers**:  
  - Dense layer with 128 neurons  
  - Output layer with 26 neurons (for 26 letters A-Z)  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
