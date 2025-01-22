import streamlit as st
from torchvision import models, transforms
from PIL import Image
import torch
import pandas as pd
import re
import time

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
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        padding: 0.5rem 2rem;
        font-size: 1.2rem;
        width: 100%;
    }
</style>
<div class="main-title">MediScan: AI-Powered Medical Image Analysis for Disease
Diagnosis</div>
<div class="description">
    Analyze medical eye images effortlessly with MediScan. <br>
    Upload patient eye images and get comprehensive diagnostic results.
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
    return label_mapping[predicted_label], confidence

def sort_eye_images(files):
    """
    Sort and pair eye images based on multiple potential naming patterns:
    1. patientID_left.jpg and patientID_right.jpg
    2. *patientID*_first_number.jpg and *patientID*_second_number.jpg
    """
    pairs = {}
    
   
    files = list(files)
    
    
    for file in files[:]:
        filename = file.name.lower()
        if '_left' in filename or '_right' in filename:
            patient_id = re.split('_left|_right', filename)[0]
            if patient_id not in pairs:
                pairs[patient_id] = {'left': None, 'right': None}
            if '_left' in filename:
                pairs[patient_id]['left'] = file
            else:
                pairs[patient_id]['right'] = file
            files.remove(file)
    
 
    while files:
        current_file = files.pop(0)
        current_filename = current_file.name.lower()
        
       
        matches = re.findall(r'_(\d+)_\d+', current_filename)
        if not matches:
            continue
        
        patient_id = matches[0]
        current_number = re.findall(r'_\d+_(\d+)', current_filename)[0]
        
        
        for other_file in files[:]:
            other_filename = other_file.name.lower()
            
          
            other_matches = re.findall(r'_(\d+)_\d+', other_filename)
            if not other_matches:
                continue
            
            other_patient_id = other_matches[0]
            other_number = re.findall(r'_\d+_(\d+)', other_filename)[0]
            
            
            if patient_id == other_patient_id:
                if patient_id not in pairs:
                    pairs[patient_id] = {'left': None, 'right': None}
                
                
                if current_number < other_number:
                    pairs[patient_id]['left'] = current_file
                    pairs[patient_id]['right'] = other_file
                else:
                    pairs[patient_id]['left'] = other_file
                    pairs[patient_id]['right'] = current_file
                
                
                files.remove(other_file)
                break
    
    return pairs

st.header("Upload Patient Eye Images")


if 'results' not in st.session_state:
    st.session_state.results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

uploaded_files = st.file_uploader(
    "Upload eye image pairs (name format: patientID_left.jpg and patientID_right.jpg)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload both left and right eye images for each patient. Name format example: 123_left.jpg and 123_right.jpg"
)

if uploaded_files:
    image_pairs = sort_eye_images(uploaded_files)
    
    if image_pairs:
        st.write("### Patient Details")
        patient_data = {}
        
        for patient_id, pair in image_pairs.items():
            if pair['left'] and pair['right']:
                st.write(f"#### Patient ID: {patient_id}")
                
                gender = st.selectbox(
                    f"Gender for patient {patient_id}",
                    ["Select Gender", "Male", "Female", "Other"],
                    key=f"gender_{patient_id}"
                )
                
               
                patient_data[patient_id] = {
                    'gender': gender,
                    'images': pair,
                }
                
                
                col1, col2 = st.columns(2)
                with col1:
                    left_image = Image.open(pair['left'])
                    left_image.thumbnail((300, 300))
                    st.image(left_image, caption=f"Left Eye - {pair['left'].name}")
                
                with col2:
                    right_image = Image.open(pair['right'])
                    right_image.thumbnail((300, 300))
                    st.image(right_image, caption=f"Right Eye - {pair['right'].name}")
                
                st.markdown("---")
        
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("Analyze Images", use_container_width=True)
        
        if analyze_button:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_patients = len(patient_data)
            for idx, (patient_id, data) in enumerate(patient_data.items(), 1):
                if data['gender'] != "Select Gender":
                    try:
                        status_text.text(f"Analyzing images for patient {patient_id}...")
                        
                       
                        left_image = Image.open(data['images']['left'])
                        right_image = Image.open(data['images']['right'])
                        
                        left_pred, left_conf = predict(left_image)
                        right_pred, right_conf = predict(right_image)
                        
                        result = {
                            "patient_id": patient_id,
                            "gender": data['gender'][0].lower(),
                            "left_eye": data['images']['left'].name,
                            "right_eye": data['images']['right'].name,
                            "left_eye_diagnosis": left_pred,
                            "left_eye_confidence": f"{left_conf:.2%}",
                            "right_eye_diagnosis": right_pred,
                            "right_eye_confidence": f"{right_conf:.2%}"
                        }
                        results.append(result)
                        
                        
                        progress = idx / total_patients
                        progress_bar.progress(progress)
                        time.sleep(0.5)  
                        
                    except Exception as e:
                        st.error(f"Error processing images for patient {patient_id}: {e}")
            
            progress_bar.progress(1.0)
            status_text.text("Analysis complete!")
            st.session_state.results = results
            st.session_state.analysis_complete = True
            time.sleep(1)  
            progress_bar.empty()
            status_text.empty()
            
           
        
        
        if st.session_state.analysis_complete and st.session_state.results:
            st.write("### Diagnosis Results")
            
            display_data = []
            download_data = []
            
            for result in st.session_state.results:
                display_data.append([
                    result["patient_id"], result["gender"], result["left_eye_diagnosis"],
                    result["left_eye_confidence"], result["right_eye_diagnosis"], result["right_eye_confidence"]
                ])
                
                download_data.append({
                    "Patient ID": result["patient_id"],
                    "Gender": result["gender"],
                    "Left Eye Diagnosis": result["left_eye_diagnosis"],
                    "Left Eye Confidence": result["left_eye_confidence"],
                    "Right Eye Diagnosis": result["right_eye_diagnosis"],
                    "Right Eye Confidence": result["right_eye_confidence"]
                })
            
            df = pd.DataFrame(display_data, columns=["Patient ID", "Gender", "Left Eye Diagnosis", "Left Eye Confidence", "Right Eye Diagnosis", "Right Eye Confidence"])
            st.dataframe(df)
            
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Diagnosis Results as CSV",
                data=csv,
                file_name="eye_disease_diagnosis_results.csv",
                mime="text/csv"
            )
    else:
        st.warning("No valid image pairs found.")
