import streamlit as st
from torchvision import models, transforms
from PIL import Image
import torch
import pandas as pd


st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #2E8B57; 
        text-align: center;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .description {
        text-align: center;
        font-size: 1.4rem;
        margin-bottom: 20px;
        color: #555555; 
        line-height: 1.6;
    }
</style>
<div class="main-title">MediScan: AI-Powered Medical Image Analysis for Disease
Diagnosis</div>
<div class="description">
    Analyze medical eye images effortlessly with MediScan. <br>
    Upload eye images, enter patient ID and gender, and let AI provide quick and accurate diagnostic results. <br>
    Download a detailed CSV report for easy record-keeping.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 4)
    model.load_state_dict(torch.load(r'C:\Users\srika\Documents\Infosys-Internship\task-1\eye_disease_modelMobileNetV2.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()


label_mapping = {
    0: "Cataract",
    1: "Diabetic Retinopathy",
    2: "Glaucoma",
    3: "Normal"
}


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  
    return image


def predict(image):
    with torch.no_grad():
        inputs = preprocess_image(image)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_label].item()
    return predicted_label, confidence


st.header("Upload and Diagnose")

uploaded_files = st.file_uploader("Upload eye images (PNG, JPG, JPEG):", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

results = []

if uploaded_files:
    st.write("### Provide Patient Details")
    for i, uploaded_file in enumerate(uploaded_files, start=1):
        try:
            image = Image.open(uploaded_file)
           
            image.thumbnail((300, 300))
            st.image(image, caption=f"Image: {uploaded_file.name}", use_container_width=False)

            
            col1, col2 = st.columns(2)
            with col1:
                patient_id = st.text_input(f"Patient ID ({uploaded_file.name})", key=f"patient_id_{uploaded_file.name}")
            with col2:
                gender = st.selectbox(f"Gender ({uploaded_file.name})", ["Male", "Female", "Other"], key=f"gender_{uploaded_file.name}")

            if patient_id and gender:
                predicted_label, confidence = predict(image)
                results.append({
                    "S.No": i,
                    "Filename": uploaded_file.name,
                    "Patient ID": patient_id,
                    "Gender": gender,
                    "Prediction": label_mapping[predicted_label],
                    "Confidence": f"{confidence:.2%}"
                })
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    
    if results:
        st.write("### Diagnosis Results")
        results_df = pd.DataFrame(results)
        results_df.index += 1  
        styled_df = results_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '14px',
            'background-color': '#f9f9f9',
            'border-color': '#dddddd'
        })
        st.dataframe(results_df, use_container_width=True)

        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="diagnosis_results.csv",
            mime="text/csv"
        )

st.markdown("""
<style>
    .footer {
        text-align: center;
        margin-top: 40px;
        font-size: 1.1rem;
        color: #888888;
        border-top: 1px solid #DDDDDD;
        padding-top: 20px;
        line-height: 1.6;
    }
    .footer strong {
        color: #2E8B57; 
    }
    .footer a {
        text-decoration: none;
        color: #4682B4; 
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
<div class="footer">
    Developed by <strong>Srikar Nulu</strong>.<br>
    Visit my <a href="https://www.linkedin.com/in/srikar-nulu-238784250/" target="_blank">LinkedIn</a> or <a href="https://github.com/srikarnulu" target="_blank">GitHub</a> for more projects.
</div>
""", unsafe_allow_html=True)   
