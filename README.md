# MediScan-Eye-Disease-Diagnosis

## **Purpose**
MediScan is an AI-powered system aimed at simplifying medical image analysis. It focuses on detecting eye diseases, enabling early diagnosis, and providing clinical decision support to improve patient care and efficiency in healthcare workflows.

---

## **Overview**
This project utilizes advanced deep learning techniques to achieve the following:

1. **Image Preprocessing**: Enhances medical image quality through noise reduction and normalization.
2. **Segmentation**: Identifies and isolates key regions like organs, tissues, or lesions for focused analysis.
3. **Feature Extraction**: Extracts important features such as texture, shape, and intensity patterns from segmented regions.
4. **Disease Classification**: Uses a lightweight pretrained MobileNetV2 model to classify images into:
   - Cataract
   - Diabetic Retinopathy
   - Glaucoma
   - Normal

The system is designed for ease of deployment in resource-constrained settings.

---

## **Prerequisites**
Make sure you have the following setup:

- **Python 3.8+**
- **Google Colab** or a system with GPU support (e.g., T4 GPU)
- **Key Python libraries**:
  - TensorFlow
  - NumPy
  - OpenCV
  - Matplotlib
  - scikit-learn
  - Streamlit (for the frontend)

---

## **Dataset**
You can download the dataset used for this project from the following link:
[Download Dataset](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

---

## **Installation**
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/MediScan-Eye-Disease-Diagnosis.git
   cd MediScan-Eye-Disease-Diagnosis
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Use a virtual environment for package management:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

---

## **Usage**

### **1. Preprocessing Images**
- Place raw eye images in the dataset directory.
- Run the preprocessing script to prepare the data:
  ```bash
  python backend.py
  ```

### **2. Training the Model**
- Ensure the preprocessed data is split into `train`, `test`, and `validation` folders.
- Start training with:
  ```bash
  python backend.py
  ```

### **3. Running the Application**
- Launch the Streamlit app for real-time predictions:
  ```bash
  streamlit run frontend.py
  ```
- Upload an eye image to classify it into one of the predefined categories.

---

## **Contributing**
Contributions are welcome! If you spot an issue or have ideas to improve the project, feel free to open an issue or submit a pull request.

---

## **License**
This project is licensed under the MIT License. For more details, see the LICENSE file.

---

## **Acknowledgments**
I would like to thank Infosys Springboard for giving me the opportunity to work on this project and extend my gratitude to the open-source community for providing the tools and frameworks that made this possible.

