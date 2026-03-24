# MLOps Unit 1 Project

## Project Overview

This project demonstrates a basic MLOps workflow using Python, Git, and machine learning.
It includes dataset analysis, model training, project organization, and reproducibility setup.

## Folder Structure

mlops-unit1/
├── data/
├── src/
│   ├── stats.py
│   └── train_model.py
├── models/
├── requirements.txt
└── README.md

## Setup Instructions

### 1. Create virtual environment

python -m venv venv

### 2. Activate environment

For Windows:
venv\Scripts\activate

### 3. Install dependencies

pip install -r requirements.txt

### 4. Run dataset statistics script

python src/stats.py

### 5. Train machine learning model

python src/train_model.py

## Output

The trained model is saved in the models folder as:
year_prediction_model.pkl
