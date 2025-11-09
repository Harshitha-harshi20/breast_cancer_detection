# Breast Cancer Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-orange)

## Project Overview
This project applies deep learning techniques to detect breast cancer from multiple imaging modalities, including **mammography, ultrasound, thermography, and MRI**. The goal is to provide an automated system to assist radiologists in accurate and early detection of breast cancer.

Key aspects of the project:
- **ResNet-50 CNN** for feature extraction
- Pretrained models with **fine-tuning**
- **Data augmentation** to improve performance
- **Class balancing** and early stopping for optimal training

---

## Features
- Multi-scan image classification
- Generates detailed **PDF reports** for predictions
- Confusion matrix generation for model evaluation
- Handles **large pretrained model files** using Git LFS
- Flask web application for interactive predictions

---

## Folder Structure
breast_cancer_detection/
│
├── models/ # Pretrained and fine-tuned models (.keras)
├── scripts/ # Python scripts for training, evaluation, and reporting
├── templates/ # HTML templates for Flask web app
├── static/ # Static assets (CSS, images, JS)
├── dataset/ # Input datasets (local only, not uploaded)
├── uploads/ # Uploaded images for prediction
├── reports/ # Generated PDF reports
├── app.py # Flask web application
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files
├── .gitattributes # Git LFS tracking
└── README.md # Project documentation

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Harshitha-harshi20/breast_cancer_detection.git
cd breast_cancer_detection
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
## Notes / Recommendations
- GPU recommended for faster predictions with TensorFlow.
- Keep datasets in dataset/ locally; do not upload to GitHub.
- .gitignore prevents temporary files, logs, and datasets from being committed.

## Authors
**Harshitha-Harshi20**  
- GitHub: [https://github.com/Harshitha-harshi20](https://github.com/Harshitha-harshi20)

## License
This project is licensed under the **MIT License**. See `LICENSE` for details.

git lfs install
