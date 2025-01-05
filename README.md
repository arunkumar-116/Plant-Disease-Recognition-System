# Plant Disease Recognition System

Welcome to the **Plant Disease Recognition System**! This project leverages deep learning to identify various plant diseases from images. Using a Convolutional Neural Network (CNN) model trained on a comprehensive dataset, this system provides accurate predictions to help protect crops and maintain healthier harvests.

## Hosted link: https://plantdiseasepredictiondemo.streamlit.app/

## Project Description

This project implements a **Plant Disease Recognition System** that enables users to upload images of plants, which are then processed and analyzed using a pre-trained machine learning model. The model identifies various plant diseases, allowing users to take timely action.

The system is built using **Streamlit** for the frontend interface, **TensorFlow** for the model, and **NumPy** for data manipulation.

## Technologies Used

- **TensorFlow**: Used for building and training the model.
- **Streamlit**: A Python framework for building the web interface.
- **NumPy**: For numerical operations.
- **Keras**: High-level neural networks API, running on top of TensorFlow.
- **Matplotlib/Seaborn**: For visualizing the confusion matrix and other metrics.

## Features

- **User-friendly Interface**: Easily upload plant images for disease prediction.
- **Real-time Prediction**: Get instant predictions about the plant’s health.
- **Model Accuracy**: The model is trained using a comprehensive plant disease dataset with multiple disease categories.
- **Visualization**: Includes the ability to visualize training and prediction results, including a confusion matrix.

## Dataset
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. The dataset consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes.

Training Set: 80% of the data used for model training.
Validation Set: 20% of the data used for validating the model.
Test Set: A new directory containing 33 test images created for prediction purposes.
Dataset source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

## Installation

### Prerequisites
Make sure you have Python 3.x installed. You can install the required dependencies using `pip`:

```bash
pip install tensorflow streamlit numpy matplotlib seaborn
```

## Usage
To run the system locally, follow these steps:

1. Start the Streamlit app:

Navigate to the project folder and run:

```bash
streamlit run main.py
```
2. Disease Recognition:

- Select the Disease Recognition page from the sidebar.
- Upload an image of a plant.
- Click Predict to get the model’s disease prediction.

## Training the Model
The model is trained using the `train_plant_disease.ipynb` notebook, which performs the following steps:

- Data Preprocessing: Loads and preprocesses plant disease images for training and validation.
- Model Architecture: A CNN model is built using multiple convolutional layers for feature extraction, followed by dense layers for classification.
- Training: The model is trained using the Adam optimizer and categorical crossentropy loss function.
- Evaluation: The model is evaluated using both training and validation datasets.
- Saving the Model: After training, the model is saved as trained_plant_disease_model.keras.
The training process took 5 epochs, with the following results:

- Epoch 1: Accuracy: 24.37%, Validation Accuracy: 67.58%
- Epoch 2: Accuracy: 68.53%, Validation Accuracy: 81.42%
- Epoch 3: Accuracy: 78.42%, Validation Accuracy: 84.50%
- Epoch 4: Accuracy: 82.24%, Validation Accuracy: 82.98%
- Epoch 5: Accuracy: 83.91%, Validation Accuracy: 86.25%

## Model Accuracy
- Training Accuracy: 83.91%
- Validation Accuracy: 86.25%

## Conclusion

The **Plant Disease Recognition System** successfully demonstrates how machine learning can be used to identify and classify plant diseases based on images. With the help of a Convolutional Neural Network (CNN), we were able to achieve impressive model accuracy, making this tool a reliable and efficient solution for plant health diagnosis.

- **Model Performance**: The model achieved an accuracy of **83.91%** on the training dataset and **86.25%** on the validation dataset, making it suitable for practical use.
- **User-Friendly Interface**: The Streamlit-based web application ensures that users can easily upload plant images and receive instant predictions about the plant’s health.
- **Future Improvements**: There is potential to improve the model's accuracy by increasing the dataset size, experimenting with more advanced architectures, or incorporating additional plant diseases.

We hope this system helps agriculturalists, gardeners, and researchers in making quicker decisions related to plant health, promoting healthier crop growth and effective disease management.

Thank you for using our system, and feel free to contribute or report any issues via the repository!

---

**Note**: Stay updated with new features by following the project on GitHub, and don't hesitate to explore the dataset further for research or training your own models.

Happy gardening! 🌱🌿🍀


