# USD/IDR Forecasting with XGBoost

A machine learning project to forecast USD/IDR exchange rate using XGBoost model.

## Project Structure

```
usd-idr-forecasting-xgboost/
├── data/
│   ├── raw/           # Raw data files
│   └── processed/    # Processed data files
├── notebooks/        # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/              # Python source code
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_model.py
│   └── predict.py
├── models/           # Trained model files
├── app/             # Streamlit application
├── reports/         # Reports and figures
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the notebooks in order
2. Train the model using `04_model_training.ipynb`
3. Evaluate using `05_model_evaluation.ipynb`
4. Run the Streamlit app: `streamlit run app/streamlit_app.py`
