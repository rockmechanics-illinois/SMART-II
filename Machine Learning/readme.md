
# Machine Learning Prediction for P-Wave and S-Wave Velocities

This repository contains all resources, datasets, scripts, and outputs required to build, evaluate, and deploy machine learning models for predicting P-Wave and S-Wave velocities. The repository is structured to ensure a streamlined workflow from preprocessing raw data to evaluating model performance.

---

## Folder Structure

### 1. **Datasets**
- **Input Data**: Contains raw input data files (e.g., LAS files, rock property measurements) organized into subfolders like `CCS1`, `CCS2`, `TRM2`, `VW1`, and `VW2`.
  - Includes `Rock Formation Depths.xlsx`, which maps depths to corresponding rock formations.
- **Training and Testing Data**: Stores preprocessed datasets used for model training and validation.


### 2. **Machine Learning Models**
- **Neural Network.ipynb**: Implements neural network models for predicting P-Wave and S-Wave velocities.
- **Random Forest.ipynb**: Includes Random Forest regression models for robust predictions.
- **Xgboost.ipynb**: Implements XGBoost models for both P-Wave and S-Wave velocities.


### 3. **Model Weights**
This folder contains pretrained model weights, allowing you to skip training and directly evaluate or use the models for prediction:
- **Neural Network for P-Wave Velocity**: Saved weights for the neural network model predicting P-Wave velocities.
- **Neural Network for S-Wave Velocity**: Saved weights for the neural network model predicting S-Wave velocities.
- **Random Forest for P-Wave Velocity**: Weights for the Random Forest model predicting P-Wave velocities.
- **Random Forest for S-Wave Velocity**: Weights for the Random Forest model predicting S-Wave velocities.
- **XGBoost for P-Wave Velocity**: Pretrained XGBoost model weights for P-Wave predictions.
- **XGBoost for S-Wave Velocity**: Pretrained XGBoost model weights for S-Wave predictions.


### 4. **Output Plots**
This directory contains visualizations for model evaluation:
- **Neural Network Plots**: Includes performance metrics and prediction visualizations for the neural network models.
- **Random Forest Plots**: Contains plots related to the performance of Random Forest models.
- **XGBoost Plots**: Visualizations and metrics generated from XGBoost models.


### 5. **Notebooks**
- **Data Processing.ipynb**:
  - Handles data cleaning, preprocessing, and feature engineering.
  - Outputs processed datasets into the `Training and Testing Data` folder.
  
- **Model Training.ipynb**:
  - Trains machine learning models (Neural Network, Random Forest, and XGBoost).
  - Evaluates models using metrics like RMSE, MAE, and \( R^2 \).

- **Understanding LAS Files.ipynb**:
  - Analyzes and interprets LAS file data.
  - Extracts and describes curve mnemonics, units, and data for rock formation analysis.

---

## Instructions to Run

### Step 1: Explore the LAS Data
- Use **`Understanding LAS Files.ipynb`** to load LAS files and understand their content and structure.
- This helps identify relevant curves for preprocessing and modeling.

### Step 2: Preprocess the Data
- Run **`Data Processing.ipynb`** to clean, normalize, and engineer features.
- Save the processed datasets into the `Training and Testing Data` folder.

### Step 3: Train the Models
- Open **`Model Training.ipynb`** and execute the cells to:
  - Train Neural Networks, Random Forest, and XGBoost models.
  - Evaluate models with various metrics and generate output plots.

### Step 4: Use Pretrained Models
- For quick predictions, load pretrained weights from the `Model Weights` folder.
- Use **`Neural Network.ipynb`**, **`Random Forest.ipynb`**, or **`Xgboost.ipynb`** to make predictions without retraining.

### Step 5: Analyze Results
- Navigate to the **Output Plots** directory to:
  - View evaluation metrics and visualizations for each model.
  - Compare model performance and refine predictions.

---

## How to Run a `.ipynb` File

1. **Install Jupyter Notebook**:
   Ensure you have Jupyter Notebook installed. You can install it using pip:
   ```bash
   pip install notebook
   ```

2. **Open the Notebook**:
   - Navigate to the folder where the notebook is stored.
   - Open the terminal or command prompt and run:
     ```bash
     jupyter notebook
     ```
   - This will open the Jupyter Notebook interface in your default web browser.

3. **Run the Cells**:
   - Click on the desired `.ipynb` file to open it.
   - Use `Shift + Enter` to execute each cell sequentially, or select **Cell > Run All** to run all cells at once.

4. **Install Required Libraries**:
   - If you encounter missing dependencies, install them using:
     ```bash
     pip install -r requirements.txt
     ```

5. **Save Outputs**:
   - Ensure you save the notebook after making any changes or running cells by clicking **File > Save and Checkpoint**.

---

## Notes
- Ensure all required Python libraries are installed:
  ```bash
  pip install -r requirements.txt
  ```
- Data files should be placed in the `Datasets/Input Data/` folder for preprocessing.
- The repository supports both training from scratch and using pretrained models for predictions.

