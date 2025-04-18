# import streamlit as st
# import torch
# import pydicom
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# from model import MyModel
#
# # 1️⃣ Streamlit page config (must be first)
# st.set_page_config(page_title="🧠 DICOM Dementia Classifier", layout="wide")
#
# # 2️⃣ Load model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MyModel()
# model.load_state_dict(torch.load("model.pth", map_location=device))
# model.to(device)
# model.eval()
#
# # 3️⃣ Image transform
# transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])
#
# # 4️⃣ App UI
# st.title("🧠 DICOM Dementia Detection")
# st.markdown("Upload a DICOM file of a patient to classify between **Vascular Dementia** and **Alzheimer’s Disease**.")
#
# uploaded_file = st.file_uploader("📤 Upload DICOM File", type=["dcm"])
#
# if uploaded_file is not None:
#     try:
#         ds = pydicom.dcmread(uploaded_file)
#
#         # Extract & display patient metadata
#         st.subheader("📋 Patient Metadata")
#         metadata = {
#             "🆔 Patient ID": getattr(ds, "PatientID", "N/A"),
#             "👤 Name": str(getattr(ds, "PatientName", "N/A")),
#             "🎂 Age": getattr(ds, "PatientAge", "N/A"),
#             "⚧️ Sex": getattr(ds, "PatientSex", "N/A"),
#             "📅 Study Date": getattr(ds, "StudyDate", "N/A"),
#             "🩻 Modality": getattr(ds, "Modality", "N/A"),
#         }
#         for key, value in metadata.items():
#             st.markdown(f"**{key}**: {value}")
#
#         # DICOM image processing
#         image_array = ds.pixel_array.astype(np.float32)
#         image_array -= np.min(image_array)
#         image_array /= np.max(image_array)
#         image_array *= 255
#         image_array = image_array.astype(np.uint8)
#         pil_image = Image.fromarray(image_array).convert("L")
#         st.image(pil_image, caption="🖼️ DICOM Image", width=300)
#
#
#         # Convert to input tensor
#         input_tensor = transform(pil_image).unsqueeze(0).to(device)
#
#         # 5️⃣ Prediction button
#         if st.button("🔍 Predict Diagnosis"):
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
#                 predicted_class = torch.argmax(output, 1).item()
#
#             label_map = {0: "🧠 Vascular Dementia", 1: "🧠 Alzheimer’s Disease"}
#             prediction = label_map[predicted_class]
#
#             # Display results in two columns
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.success(f"🧪 **Diagnosis:** {prediction}")
#             with col2:
#                 st.info("📊 **Model Confidence**")
#                 st.write(f"Vascular Dementia: {probs[0]*100:.2f}%")
#                 st.write(f"Alzheimer’s Disease: {probs[1]*100:.2f}%")
#
#     except Exception as e:
#         st.error("❌ This doesn't seem like a valid DICOM file.")
#         st.exception(e)

 #2
# import streamlit as st
# import os
# import torch
# import pydicom
# import numpy as np
# import cv2
# from PIL import Image
# from torchvision import transforms
#
# # 🚀 Set page config
# st.set_page_config(page_title="DICOM Folder Diagnosis", layout="wide")
#
# # 🧠 Model Definition
# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
#         self.fc2 = torch.nn.Linear(128, 2)
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 32 * 32 * 32)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
# @st.cache_resource
# def load_model():
#     model = MyModel()
#     model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
#     model.eval()
#     return model
#
# model = load_model()
#
# # 🧾 Title and Instructions
# st.title("🧠 DICOM MRI Folder Diagnosis")
# st.markdown("Upload a folder of DICOM slices (e.g. MRI) for a single patient.")
#
# # 📂 Upload DICOM Folder
# uploaded_files = st.file_uploader("Upload DICOM Folder", type=["dcm"], accept_multiple_files=True)
#
# if uploaded_files:
#     # Sort files by instance number if available
#     dicom_files = []
#     for file in uploaded_files:
#         try:
#             dcm = pydicom.dcmread(file)
#             instance_num = getattr(dcm, "InstanceNumber", 0)
#             dicom_files.append((instance_num, dcm))
#         except:
#             continue
#
#     dicom_files.sort(key=lambda x: x[0])
#     slices = [dcm.pixel_array for _, dcm in dicom_files]
#
#     if not slices:
#         st.error("❌ No valid DICOM slices found.")
#     else:
#         # 🧠 Patient Info from first slice
#         st.subheader("🧾 Patient Info")
#         dcm = dicom_files[0][1]
#         patient_info = {
#             "Patient ID": getattr(dcm, "PatientID", "N/A"),
#             "Patient Name": getattr(dcm, "PatientName", "N/A"),
#             "Sex": getattr(dcm, "PatientSex", "N/A"),
#             "Age": getattr(dcm, "PatientAge", "N/A"),
#             "Study Date": getattr(dcm, "StudyDate", "N/A"),
#             "Modality": getattr(dcm, "Modality", "N/A"),
#         }
#
#         # 🧩 Layout to display image and patient info side by side
#         col1, col2 = st.columns([2, 3])
#         with col1:
#             st.markdown("### 🧾 Patient Information")
#             for key, value in patient_info.items():
#                 st.write(f"**{key}:** {value}")
#
#         # 🖼️ Show middle slice
#         with col2:
#             st.subheader("🧩 MRI Scan (Middle Slice)")
#             mid_index = len(slices) // 2
#             mid_slice = slices[mid_index]
#             normalized = cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#             st.image(normalized, caption="Middle Slice", width=300)
#
#         # 🧠 Predict button
#         st.subheader("🔍 Diagnosis")
#         if st.button("Predict"):
#             transform = transforms.Compose([
#                 transforms.ToPILImage(),
#                 transforms.Resize((128, 128)),
#                 transforms.Grayscale(num_output_channels=1),
#                 transforms.ToTensor()
#             ])
#
#             input_tensor = transform(normalized).unsqueeze(0)
#             with torch.no_grad():
#                 output = model(input_tensor)
#                 predicted_class = torch.argmax(output, dim=1).item()
#                 confidence = torch.softmax(output, dim=1).numpy().flatten()
#
#             label_map = {0: "Vascular Dementia", 1: "Alzheimer’s Disease"}
#             prediction_text = label_map[predicted_class]
#
#             st.success(f"**✅ Diagnosis: {prediction_text}**")
#             st.write(f"🧮 Confidence: Vascular = `{confidence[0]:.2f}`, Alzheimer = `{confidence[1]:.2f}`")
#
# else:
#     st.info("Upload a folder of DICOM files to begin.")
#


#3

import streamlit as st
import os
import torch
import pydicom
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# 🚀 Set page config
st.set_page_config(page_title="DICOM Folder Diagnosis", layout="wide")

# 🧠 Model Definition
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_model():
    model = MyModel()
    model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# 🧾 Title and Instructions
st.title("🧠 DICOM MRI Folder Diagnosis")
st.markdown("Upload a folder of DICOM slices (e.g. MRI) for a single patient.")

# 📂 Upload DICOM Folder
uploaded_files = st.file_uploader("Upload DICOM Folder", type=["dcm"], accept_multiple_files=True)

if uploaded_files:
    # Sort files by instance number if available
    dicom_files = []

    for file in uploaded_files:
        try:
            dcm = pydicom.dcmread(file)
            if 'PixelData' not in dcm:
                continue

            instance_num = getattr(dcm, "InstanceNumber", 0)
            pixel_array = dcm.pixel_array

            # Handle multi-frame grayscale (e.g. shape: [N, H, W])
            if pixel_array.ndim == 3:
                if pixel_array.shape[2] == 3:  # (H, W, 3)
                    # Single color image with 3 channels
                    dicom_files.append((instance_num, pixel_array, dcm))
                else:
                    # Multiple grayscale slices (e.g. shape [N, H, W])
                    for i, frame in enumerate(pixel_array):
                        dicom_files.append((instance_num * 1000 + i, frame, dcm))
            else:
                dicom_files.append((instance_num, pixel_array, dcm))

        except Exception as e:
            pass  # Skip files that can't be read


    # Sort by instance/frame number
    dicom_files.sort(key=lambda x: x[0])
    slices = [arr for _, arr, _ in dicom_files]


    dicom_files.sort(key=lambda x: x[0])


    if not slices:
        st.error("❌ No valid DICOM slices with image data found.")
    else:
        # 🧠 Patient Info from first slice
        st.subheader("🧾 Patient Info")
        dcm = dicom_files[0][2]  # Third item is the original DICOM object

        patient_info = {
            "Patient ID": getattr(dcm, "PatientID", "N/A"),
            "Patient Name": getattr(dcm, "PatientName", "N/A"),
            "Sex": getattr(dcm, "PatientSex", "N/A"),
            "Age": getattr(dcm, "PatientAge", "N/A"),
            "Study Date": getattr(dcm, "StudyDate", "N/A"),
            "Modality": getattr(dcm, "Modality", "N/A"),
        }

        # 🧩 Layout to display image and patient info side by side
        col1, col2 = st.columns([2, 3])
        with col1:
            st.markdown("### 🧾 Patient Information")
            for key, value in patient_info.items():
                st.write(f"**{key}:** {value}")

        # 🖼️ Show middle slice
        # 🖼️ Show middle slice
        with col2:
            st.subheader("🧩 MRI Scan (Middle Slice)")
            mid_index = len(slices) // 2
            mid_slice = slices[mid_index]

            # Normalize and convert for display
            if isinstance(mid_slice, np.ndarray):
                if mid_slice.ndim == 2:
                    normalized = cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    st.image(normalized, caption="Middle Slice (Grayscale)", width=300, channels="L")
                elif mid_slice.ndim == 3 and mid_slice.shape[2] == 3:
                    normalized = cv2.normalize(mid_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    st.image(normalized, caption="Middle Slice (RGB)", width=300, channels="RGB")
                else:
                    st.warning("⚠️ Unsupported image shape for display.")
            else:
                st.error("❌ Could not convert middle slice to an image.")

        # 🧠 Predict button
        st.subheader("🔍 Diagnosis")
        if st.button("Predict"):
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])

            input_tensor = transform(normalized).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1).numpy().flatten()

            label_map = {0: "Vascular Dementia", 1: "Alzheimer’s Disease"}
            prediction_text = label_map[predicted_class]

            st.success(f"**✅ Diagnosis: {prediction_text}**")
            st.write(f"🧮 Confidence: Vascular = `{confidence[0]:.2f}`, Alzheimer = `{confidence[1]:.2f}`")

else:
    st.info("Upload a folder of DICOM files to begin.")
