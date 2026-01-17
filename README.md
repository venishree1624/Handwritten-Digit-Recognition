# Handwritten Digit Recognition using Machine Learning

## 📌 Project Overview
This project focuses on building a machine learning-based system to recognize handwritten digits (0–9) using Python.  
The model is trained on a standard handwritten digit dataset and is capable of predicting digit labels with high accuracy.

This project was developed as part of the **Microsoft Elevate – Azure Fundamentals Internship Program**, with the objective of understanding machine learning workflows and their relevance to cloud-based AI solutions.

---

## 🎯 Problem Statement
Manual recognition of handwritten digits is time-consuming and prone to errors, especially when processing large volumes of data such as forms, documents, and exam papers.  
An automated system is required to accurately identify handwritten digits and convert them into machine-readable format.

---

## 💡 Solution Overview
A supervised machine learning approach is used to train a classification model that learns patterns from handwritten digit images and predicts the corresponding digit labels accurately.

---

## 🛠️ Technologies Used
- Python 3.10
- NumPy
- Matplotlib
- Scikit-learn

---

## ⚙️ Algorithm Used
- **K-Nearest Neighbors (KNN) Classification**

The KNN algorithm was selected due to its simplicity and effectiveness in image-based classification tasks.

---

## 📊 Dataset
- Built-in handwritten digit dataset from **Scikit-learn**
- Image size: 8×8 grayscale pixels
- Digit classes: 0 to 9

This dataset is commonly used for educational and machine learning demonstration purposes.

---

## 🚀 How to Run the Project
1. Install Python 3.10
2. Install required libraries:
pip install numpy matplotlib scikit-learn
3. Run the Python file:
python digit_recognition.py

---

## ✅ Results
- Model accuracy achieved: **~98%**
- Correct prediction of multiple handwritten digits
- Output images displayed with predicted labels
- The model was tested with different digit samples (e.g., 0 and 5), proving consistent performance

---

## ☁️ Azure Fundamentals Relevance
This project aligns with **Azure AI & Machine Learning fundamentals** covered under the Microsoft Elevate internship program.

Although the model is implemented locally, it follows a complete machine learning workflow (data loading, training, testing, and evaluation) that can be extended to cloud platforms.

In the future, this model can be:
- Deployed using **Microsoft Azure Machine Learning**
- Hosted on **Azure Virtual Machines**
- Integrated into cloud-based AI applications

---

## 🔮 Future Scope
- Test the model with real-world handwritten digit images
- Improve accuracy using deep learning techniques such as CNN
- Deploy the model on Microsoft Azure for scalable cloud inference
- Integrate the system into web or mobile applications

---

## 👩‍💻 Author
**Venishree T**

Developed as part of the  
**Microsoft Elevate – Azure Fundamentals Internship Program**

---
