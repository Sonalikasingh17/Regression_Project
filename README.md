# Laptop Price Predictor

A machine learning project to predict the price of laptops based on various specifications like brand, processor, RAM, storage, and operating system using regression models.

---

## Project Structure

- laptop-price-predictor.ipynb # Main notebook for data analysis and modeling
- dataset(laptop_price) # Folder for dataset CSV file
- a python file (price_prdictor)
- README.md # Project documentation
- requirements.txt # Python dependencies
- models/ # Saved models 

---

##  Dataset

The dataset includes details like:

- Company
- Product
- TypeName
- RAM
- Weight
- Touchscreen
- IPS
- Screen Size
- Resolution
- CPU
- HDD
- SSD
- GPU
- OS
- Price

The target variable is `Price` (in INR or any relevant currency).

---

## Key Steps

### 1. Data Preprocessing
- Cleaned missing values
- Converted categorical data using encoding
- Feature extraction (e.g., screen resolution to PPI)
- Normalized numerical data

### 2. Feature Engineering
- Binary encoding for touchscreen and IPS display
- Combined `Weight`, `PPI`, and other derived metrics
- One-hot encoding for categorical variables

### 3. Model Building
- Tried multiple regression models including:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Random Forest Regressor

- Used pipelines for cleaner implementation.

### 4. Model Evaluation
- Evaluated using:
  - R² Score
  - MAE/MSE/RMSE
- Performed hyperparameter tuning (GridSearchCV)

---

##  Model Architecture

The pipeline includes:

ColumnTransformer [
(numerical_pipeline): StandardScaler
(categorical_pipeline): OneHotEncoder
] → Regressor (e.g., RandomForest)


---

##  Evaluation Metrics
```yaml
- **R² Score**: Indicates the proportion of variance explained
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
```
---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
## Sample requirements.txt:

```nginx
numpy
pandas
scikit-learn
matplotlib
seaborn
```
