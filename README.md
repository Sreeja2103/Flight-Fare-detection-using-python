# Flight-Fare-detection-using-python
Flight Price Prediction

This project uses machine learning to predict flight ticket prices based on historical data. It is implemented in a Jupyter Notebook (main.ipynb) using scikit-learn's Random Forest Regressor along with hyperparameter tuning.

Features

Loads and processes flight price dataset.

Trains a machine learning model with Random Forest Regressor.

Splits data into training and testing sets.

Evaluates model performance using:

R² (Coefficient of Determination)

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

RMSE (Root Mean Squared Error)

Hyperparameter tuning with RandomizedSearchCV.

Visualization of actual vs predicted prices.

Requirements

Make sure you have the following Python libraries installed:

pip install pandas scikit-learn matplotlib scipy

Dataset

The project uses a dataset named Clean_Dataset.csv.zip (flight pricing data). Update the path to the dataset in the notebook if needed.

Usage

Clone this repository or download the files.

Ensure the dataset is placed in the correct path.

Open the Jupyter Notebook:

jupyter notebook main.ipynb

Run the cells step by step to:

Load the dataset

Train the model

Evaluate performance

Visualize predictions

Project Structure

├── main.ipynb              # Jupyter Notebook containing the full workflow
├── Clean_Dataset.csv.zip   # Flight pricing dataset (not included in repo)
└── README.md               # Project documentation

Example Output

Performance metrics printed in the notebook.

Scatter plot of Actual vs Predicted Flight Prices.

Author

Developed as a machine learning project for flight price prediction.

