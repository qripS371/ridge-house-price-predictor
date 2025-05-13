# Ridge Regression House Price Predictor

A Python-based CLI application that predicts house prices using Ridge Regression on a multivariate dataset. It includes full preprocessing with Pandas, modeling with scikit-learn, and performance visualization with Matplotlib. Achieves an R² score of ~0.84, offering a practical baseline for regularized regression problems.

---

## Features

- Ridge Regression model with regularization
- Clean data preprocessing pipeline
- R² score of ~0.84
- CLI support for interactive predictions
- Visualizations for insights

---

## Installation

```bash
git clone https://github.com/qripS371/ridge-house-price-predictor
cd ridge-house-price-predictor
pip install -r requirements.txt
Usage
bash
Copy
Edit
python house_price_prediction.py
Ensure train.csv is in the same directory as the script.

Requirements
Python 3.x

pandas

scikit-learn

matplotlib

You can install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Dataset
The model uses a housing dataset (train.csv) containing features like:

Lot area

Year built

Number of rooms

Neighborhood info

etc.

Make sure train.csv is present in the root folder before running the script.

Project Structure
Copy
Edit
.
├── house_price_prediction.py
├── train.csv
├── requirements.txt
└── README.md
License
MIT License

Author
Prithvi
GitHub: qripS371

yaml
Copy
Edit

---

Let me know if you want me to create a `requirements.txt` or `.gitignore` file next!







