#DiabetesPredictionModel

⚕️ Diabetes Prediction Model

This project features a machine learning model built with a Decision Tree Classifier to predict the likelihood of a person having diabetes. The model is trained on a comprehensive dataset of medical records, and the included Jupyter Notebook allows for both model training and live prediction based on user input.

🧐 Overview

The core of this project is a machine learning pipeline that takes eight key health metrics as input:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

The trained model then classifies the person as either diabetic or non-diabetic. The notebook also includes code for evaluating the model's performance using a confusion matrix and a classification report.

🚀 Getting Started

Prerequisites
Python 3.x

Jupyter Notebook or a similar environment to run the .ipynb file.

Installation

Clone the repository:

Bash

git clone https://github.com/your-username/diabetes-prediction-model.git
cd diabetes-prediction-model
Install the required libraries:

Bash

pip install pandas numpy scikit-learn
Usage
Ensure both diabetes.csv and the diabetes.ipynb notebook are in the same directory.

Open and run the diabetes.ipynb notebook in your environment.

The notebook will guide you through the data loading, model training, and evaluation steps.

At the end, it will prompt you to enter the health metric values for a person you want to analyze.

After entering the values, the program will output its prediction.

Example
Here's a sample of the prediction output after the user enters the required values:

Enter pregnancies value:1
Enter glucose value:130
Enter blood pressure value:70
Enter skin thickness value:30
Enter insulin value:100
Enter bmi value:35.0
Enter diabetes pedigree function value:0.5
Enter age:30
The prediction: [0]
The person is not diabetic
📁 Project Structure
diabetes.ipynb: The main Jupyter Notebook containing all the code for data processing, model training, evaluation, and user-input predictions.

diabetes.csv: The dataset used to train the machine learning model.

README.md: This file.

🛠️ Built With
Pandas - For data manipulation and analysis.

Numpy - For numerical operations.

Scikit-learn - For building the Decision Tree Classifier and evaluating the model.

✍️ Authors
Suriya Panneerselvam
