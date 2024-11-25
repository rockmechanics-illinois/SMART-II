# SMART-II

This repository contains data and machine learning workflows for analyzing rock properties in the Illinois Basin Decatur Project (IBDP). It is structured to include geomechanical data, initial exploratory work, and machine learning models to predict rock property parameters.

---

## Repository Structure

### **1. Geomechanics Summary**
- `all data (core log).xlsx`: Core log data summarizing rock properties.
- `Illinois Basin - Summary of measured rock properties.xlsx`: Detailed summary of measured rock properties in the Illinois Basin.

### **2. Initial Work**
Contains early notebooks used for exploratory analysis and initial modeling:
- `log TR Decision Trees.ipynb`
- `log TR Gradient Boost.ipynb`
- `log TR Linear Regression.ipynb`
- `log TR Neural Network (3 layers).ipynb`
- `min_log.csv`: Required dataset for the notebooks.

### **3. Machine Learning**
- **Datasets**:
  - `Input Data`: Contains raw input data files.
  - `Training and Testing data`: Contains preprocessed datasets for training and testing models.
- **Model Weights**:
  - `best_rf_model.joblib`: Saved weights for Random Forest.
  - `best_xgb_model.joblib`: Saved weights for XGBoost.
  - `best1_xgb_model.joblib`: Alternative XGBoost model weights.
- **Outputs**:
  - `Neural Network Plots`: Contains plots generated during Neural Network training.
  - `Random Forest Plots`: Contains plots generated during Random Forest training.
  - `XGBoost Plots`: Contains plots generated during XGBoost training.
- **Notebooks**:
  - `Data Processing.ipynb`: Handles preprocessing of the data.
  - `Model Building.ipynb`: Trains and evaluates Random Forest, XGBoost, and Neural Network models.
  - `Model Run.ipynb`: A simplified version focused on XGBoost for quick evaluation.
  - `Understanding LAS Files.ipynb`: Analyzes LAS file data to extract and interpret curve information.

---

## How to Run

### Step 1: Download Required Data
Ensure you download the `min_log.csv` file and place it in the repository's root directory.

### Step 2: Set Up the Environment
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Notebooks
Simply open and run the Jupyter notebooks in the `Machine Learning` folder using your preferred environment (VSCode, Jupyter Lab, or localhost).

---

## Notes
- Use `Data Processing.ipynb` for data preprocessing.
- Use `Model Building.ipynb` for comprehensive training and evaluation of all models.
- Use `Model Run.ipynb` for quick evaluation using XGBoost.

