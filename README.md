# 🧪 Sickle Cell Disease Detector & Visual Question Answering (VQA)

This AI-powered medical tool detects **sickle cell disease** from blood smear images and answers natural-language questions related to the disease. It combines **image classification**, **cell highlighting**, and **medical question-answering** into one powerful diagnostic interface.

---

## 📌 Key Features

✅ Classifies blood smear images as:
- **Sickle Cell**
- **Normal Cell**

✅ Highlights **abnormal sickle-shaped cells** in the image using image processing.

✅ Accepts **natural-language questions** like:
- *“Is it inherited?”*
- *“What are the symptoms?”*
- *“Can I travel with sickle cell?”*
- *“Show me sickle cells in this image”*

✅ Provides **personalized, medical-grade answers** using AI-powered VQA logic.

---

## 🧠 Model Training (DenseNet121)

We used **DenseNet121**, a deep convolutional neural network pretrained on **ImageNet**, and fine-tuned it on labeled blood smear images.

- Input size: `224x224`
- Classification: Binary
  - `1` → Sickle Cell
  - `0` → Normal Cell
- Framework: **TensorFlow / Keras**
- Final model saved as: `model_fold5.h5`

> DenseNet121 was chosen for its efficiency and ability to retain detailed features in medical images.

---

## 🔬 Image Analysis

- Uploaded images are processed through the trained DenseNet121 model
- Model returns:
  - **Predicted label**
  - **Confidence score**
- Sickle cells are visually highlighted using OpenCV:
  - **Red boxes** → Severe sickling
  - **Orange boxes** → Moderate sickling

---

## 🤖 Visual Question Answering (VQA)

Users can ask **natural-language questions**, and the app responds based on:
- The uploaded image
- Model prediction
- Question intent

VQA covers 40+ question types, such as:
- Symptoms
- Cure options
- Inheritance
- Pain crisis
- Diet
- Travel
- Growth/fertility
- Early signs in children
- Treatment risks (temporary & permanent)
- Bone marrow transplant
- Ayurvedic remedies
- …and many more!

---

## ⚙️ Technologies Used

| Component            | Tech Stack                    |
|----------------------|-------------------------------|
| Model Architecture   | DenseNet121 (Transfer Learning) |
| Framework            | TensorFlow / Keras            |
| UI                   | Streamlit                     |
| Natural Language QA  | Custom NLP logic using Python |
| Visualization        | Matplotlib                    |

---



