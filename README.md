# SMART-II

This repository contains data and machine learning workflows for analyzing rock properties in the Illinois Basin Decatur Project (IBDP). It is structured to include geomechanical data, initial exploratory work, and machine learning models to predict rock property parameters. This repository contains all resources, scripts, and data for analyzing and predicting geomechanical properties such as **P-Wave Velocity** and **S-Wave Velocity** based on mineralogical and other rock properties. The work includes datasets, initial explorations, machine learning models, and outputs.

---

## Directory Structure

### 1. **Geomechanics Summary**
   - **all data (core log).xlsx**: Comprehensive core log data containing measured rock properties.
   - **Illinois Basin - Summary of measured rock properties.xlsx**: Dataset summarizing rock properties specific to the Illinois Basin.

### 2. **Initial Work**
   - **log TR Decision Trees.ipynb**: Notebook exploring decision trees for initial velocity predictions.
   - **log TR Gradient Boost.ipynb**: Gradient boosting regression experiments for velocity predictions.
   - **log TR Linear Regression.ipynb**: Baseline linear regression models for P-Wave and S-Wave velocities.
   - **log TR Neural Network (3 layers).ipynb**: Initial trials with a three-layer neural network.
   - **min_log.csv**: Processed data for early trials.

### 3. **Machine Learning**
   - **Datasets**:
     - **Input Data**: Raw data used for modeling.
     - **Training and Testing data**: Preprocessed datasets for model training and testing.
   - **Machine Learning Models**:
     - **Neural Network.ipynb**: Fitting Neural network models using the weights for P-Wave and S-Wave velocity predictions.
     - **Random Forest.ipynb**: Fitting Random forest models using the weights for P-Wave and S-Wave velocity predictions.
     - **Xgboost.ipynb**: Fitting XGBoost models using the weights for P-Wave and S-Wave velocity predictions.
   - **Model weights**:
     - Contains ML model weights and saved models for:
       - **Neural Network for P-Wave and S-Wave Velocity**.
       - **Random Forest for P-Wave and S-Wave Velocity**.
       - **XGBoost for P-Wave and S-Wave Velocity**.

### 4. **Output Plots**
   - **Neural Network Plots**: Visualization of results from neural network models.
   - **Random Forest Plots**: Output plots for random forest predictions.
   - **XGBoost Plots**: Results and performance metrics visualized for XGBoost.

### 5. **Additional Notebooks**
   - **Data Processing.ipynb**: Code for data cleaning, preprocessing and feature engineering for the datasets.
   - **Model Training.ipynb**: Training workflows for all the machine learning models.
   - **Understanding LAS Files.ipynb**: Helper notebook to parse and understand LAS files.

---

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Jupyter Notebook
- Required libraries listed in `requirements.txt`.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/geomechanics-summary.git
   ```
2. Navigate to the repository directory:
   ```bash
   cd geomechanics-summary
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Navigating the Code

### 1. **Explore the Data**:
  - The "Understanding LAS Files.ipynb" notebook is designed to read and explore LAS (Log ASCII Standard) files, commonly used for geophysical and geological data. It provides detailed insights into the structure and content of LAS files by extracting metadata for each curve (log), including its mnemonic, unit, description, and the number of data points. Using the `lasio` library, the notebook analyzes specific LAS files, such as those containing gamma ray logs, sonic travel time, and mechanical property data, helping users identify relevant curves for downstream processing and modeling. This step ensures a clear understanding of the data's structure and quality before further analysis.
    - Python Notebook to run: Understanding LAS Files.ipynb
    - Path: ./Machine Learning/
    - Imports from: ./Machine Learning/Datasets/Input Data/

### 2. **Data Processing**:
   - The `Data Processing.ipynb` notebook handles the cleaning and preprocessing of raw datasets to prepare them for machine learning workflows. It performs tasks such as handling missing values, applying Min-Max scaling for normalization, and engineering features where needed. This ensures that the data is consistent, structured, and ready for training. The notebook outputs the cleaned datasets in CSV format, which can then be used for further analysis and modeling.
     - **Python Notebook to run**: Data Processing.ipynb  
     - **Path**: ./Machine Learning/  
     - **Imports from**: ./Machine Learning/Datasets/Input Data/, Rock Formation Depths.xlsx
     - **Exports to:**  ./Machine Learning/Datasets/Training and Testing data

### 3. **Model Building and Training**:
   - The `Model Training.ipynb` notebook is used to define machine learning pipelines for various models, such as Neural Networks, Random Forest, and XGBoost. It allows users to configure hyperparameters and incorporate preprocessing techniques like PCA or undersampling. Once the models are designed, the notebook executes the training workflow, evaluates the models using metrics such as MAE, RMSE and \(R^2\), and saves the trained models for reproducibility. These notebooks provide a comprehensive workflow for creating robust predictive models.
     - **Python Notebooks to run**: Model Training.ipynb  
     - **Path**: ./Machine Learning/  
     - **Imports from**: ./Machine Learning/Input Data/  
     - **Exports to**: ./Machine Learning/Model Weights/ 

### 4. **Model Evaluation and Assessment (without Training)**:
   - The notebooks in the **Machine Learning Models** directory allow users to evaluate and assess model performance without retraining. These notebooks load pretrained model weights (e.g., for Neural Networks, Random Forest, and XGBoost) and use them to predict outcomes based on new input data. This enables quick assessments and predictions without the computational overhead of training. Users can analyze the predicted results alongside true values and generate evaluation metrics to measure the models' accuracy and reliability.  
     - **Python Notebooks to run**: Notebooks in the **Machine Learning Models** directory: `Xgboost.ipynb`, `Random Forest.ipynb`, `Neural Network.ipynb` 
     - **Path**: ./Machine Learning/Machine Learning Models/  
     - **Imports from**: ./Machine Learning/Input Data/, ./Machine Learning/Model Weights/  

### 5. **Visualizing Results**:
   - The **Output Plots** directory contains visualizations generated from the trained machine learning models to evaluate their performance. These include loss curves, predicted vs. actual values, feature importance plots, and residual analysis. By examining these plots, users can assess model accuracy, identify trends, and pinpoint areas for improvement. The visualizations are essential for understanding how well the models are performing and for fine-tuning them further.  
     - **Path**: ./Machine Learning/Output Plots/  
     - **Outputs from**: Model Training.ipynb  

---

## Results

The repository includes:
- Comparisons of machine learning models (Random Forest, XGBoost, Neural Networks).
- Pretrained weights for reproducibility.
- Output visualizations showcasing model performance.

**For More Info kindly explore the readme in the Machine Learning Directory.