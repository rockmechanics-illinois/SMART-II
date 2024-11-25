# Machine Learning Prediction for P and S Wave velocities

This folder contains all resources, scripts, and outputs related to building and evaluating machine learning models for the project. The structure is organized into datasets, model weights, and notebooks for different stages of the pipeline.

---

## Folder Structure

### 1. **Datasets**
- **Input Data**: Contains raw input data files used for training and testing the models.
- **Training and Testing Data**: Contains preprocessed datasets used for model training and validation.

### 2. **Model Weights**
- `best_xgb_model.joblib`: Saved weights for the best-performing XGBoost model for P wave velocity prediction.
- `best1_xgb_model.joblib`: Saved weights for an additional XGBoost model configuration for S wave velocy prediction.

### 3. **Outputs**
- **Neural Network Plots**: Contains plots generated during Neural Network training and evaluation.
- **Random Forest Plots**: Contains plots generated during Random Forest training and evaluation.
- **XGBoost Plots**: Contains plots generated during XGBoost training and evaluation.

### 4. **Notebooks**
- **Data Processing.ipynb**:
  Handles data preprocessing, including scaling, splitting, and feature engineering for training and testing datasets.
  
- **Model Building.ipynb**:
  Includes all models:
  - Random Forest
  - XGBoost
  - Neural Network
  This notebook trains, evaluates, and compares these models using appropriate metrics (MAE, RMSE, R²).

- **Model Run.ipynb**:
  A streamlined version of `Model Building.ipynb` focused only on XGBoost. It provides a quick way to train and evaluate the XGBoost model.

- **Understanding LAS Files.ipynb**:
  Helps analyze and interpret LAS file data, including extraction and understanding of curve mnemonics and descriptions.

---

## Instructions to Run

### Step 1: Preprocess the Data
1. Open the **`Data Processing.ipynb`** notebook.
2. Run all cells to preprocess the data and save the training and testing datasets into the appropriate folders.

### Step 2: Train All Models
1. Open the **`Model Building.ipynb`** notebook.
2. Run all cells to train Random Forest, XGBoost, and Neural Network models.
3. Evaluate and compare the models using the generated metrics and plots.

### Step 3: Run XGBoost Only
1. Open the **`Model Run.ipynb`** notebook.
2. Run all cells to train and evaluate the XGBoost model.
3. The output includes metrics, predictions, and plots for XGBoost.

### Step 4: Analyze Outputs
- Navigate to the **Outputs** folder to view:
  - Plots for Neural Network, Random Forest, and XGBoost models.
  - Predictions and metrics for each model.

---

## Notes
- Ensure the required Python libraries are installed:
  ```
  pip install -r requirements.txt
  ```
- The `Model Weights` folder contains pretrained models that can be loaded to skip training and directly generate predictions.
- For LAS file analysis, use the **`Understanding LAS Files.ipynb`** notebook.
