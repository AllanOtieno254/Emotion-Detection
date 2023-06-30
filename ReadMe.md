# Facial Expression Detection

This Python program detects the location of a face in an input image or frame and classifies the emotion on the face. It utilizes machine learning algorithms, such as Convolutional Neural Networks (CNNs), to perform facial expression recognition.

## Requirements

- Python 3.x
- OpenCV (cv2) library
- TensorFlow library
- Keras library

## Installation

1. Clone or download the project repository from [GitHub](https://github.com/TheODDYSEY/Emotion-Detect-OpenCV.git).

2. Install the required libraries using the following command:
 
 pip install opencv-python

 pip install tensorflow


pip install keras




3. Download the pre-trained Haar cascade classifier XML file for face detection from the OpenCV GitHub repository. Place the XML file in the `Harcascade` directory.

4. Download the pre-trained emotion classification model file (in HDF5 format) and place it in the `Models` directory.

## Usage

1. In the Python script `Emotion_Detection.py`, set the `image_path` variable to the path of the input image or frame you want to analyze.

2. Run the Python script using the following command:

         python Emotion_Detection.py


3. Download the pre-trained Haar cascade classifier XML file for face detection from the OpenCV GitHub repository. Place the XML file in the `Harcascade` directory.

4. Download the pre-trained emotion classification model file (in HDF5 format) and place it in the `Models` directory.

## Usage

1. In the Python script `Emotion_Detection.py`, set the `image_path` variable to the path of the input image or frame you want to analyze.

2. Run the Python script using the following command: 

3. The program will load the input image, detect faces using the Haar cascade classifier, draw bounding boxes around the faces, and classify the emotions on the faces.

4. The output image with bounding boxes and emotion labels will be displayed. Press any key to close the image window.

## Customization

- You can modify the class labels in the `class_labels` dictionary to match your specific emotion classification labels.

- Adjust the parameters of the `face_classifier.detectMultiScale()` function to change the face detection sensitivity.

## References

- [OpenCV documentation](https://docs.opencv.org/)
- [TensorFlow documentation](https://www.tensorflow.org/api_docs)
- [Keras documentation](https://keras.io/api/)
- "A Convolutional Neural Network Cascade for Face Detection" by R. Lienhart and J. Maydt: [ResearchGate](https://www.researchgate.net/publication/3940582_Rapid_Object_Detection_using_a_Boosted_Cascade_of_Simple_Features)
- "Facial Expression Recognition Using Convolutional Neural Networks: State of the Art" by I. Barros and J. Y. Kam: [arXiv](https://arxiv.org/abs/1612.02903)
- Cascade Classifier: [OpenCV documentation](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- Face Detection with Haar Cascade: [Towards Data Science](https://towardsdatascience.com/face-detection-with-haar-cascade-727f68dafd08)

## Disclaimer

The accuracy of emotion classification depends on the quality of the pre-trained model and the input images or frames.


