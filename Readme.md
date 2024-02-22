# Hybrid Model with Neural Network and Random Forest for Fault Detection in Tennessee Eastman Process Simulation Dataset

## Introduction

This project aims to develop a hybrid model combining a neural network (NN) and a random forest (RF) classifier for fault detection in the Tennessee Eastman Process Simulation dataset. The NN extracts features, which are then fed into the RF classifier for fault classification. This approach leverages the strengths of both models to enhance fault detection accuracy.

### Steps Taken 🚀

1. **Importing Libraries**: Essential libraries including PyReadr for reading RData files, Matplotlib for visualization, Seaborn for statistical graphics, NumPy, Pandas for data manipulation, and Keras for building deep learning models were imported.
2. **Importing Dataset**: Fault-free and faulty training datasets were located and imported. PyReadr was used to read RData files, and Pandas for data manipulation.
3. **Preprocessing Dataset**: Data preprocessing involved filtering, scaling, and encoding categorical variables using StandardScaler and OneHotEncoder from scikit-learn.
4. **Preparing Neural Network**: A neural network model was constructed with dense layers for feature extraction and classification.
5. **Training Neural Network**: The neural network model was trained on the preprocessed data with early stopping to prevent overfitting.
6. **Feature Extraction with Intermediate Layer**: An intermediate layer of the neural network was extracted to obtain feature representations.
7. **Training Random Forest Classifier**: The extracted features were used to train a random forest classifier.
8. **Evaluating the Model**: The performance of the combined NN + RF model was evaluated using confusion matrices and accuracy scores.

## Results

- **Training Performance**: The training history depicted decreasing loss and increasing accuracy over epochs, indicating effective learning.
- **Evaluation Metrics**: The confusion matrix revealed the model's ability to accurately predict fault classes, with an overall accuracy score computed.
- **Real-time Fault Prediction**: Visualization of the model's predictions for real-time fault detection demonstrated its capability to identify fault classes over time.
- **Overall Accuracy**: The overall accuracy of the model was calculated as 93.7%, indicating high performance across fault classes.

## Conclusion

This project demonstrates the effectiveness of a hybrid model combining a neural network and random forest classifier for fault detection in industrial processes using the Tennessee Eastman Process Simulation dataset. By leveraging both deep learning and traditional machine learning techniques, the model achieves high accuracy in classifying fault instances, showcasing its potential for real-world application in process monitoring and control systems.
