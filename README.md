# 🚀 MiniML-Toolkit — Machine Learning Using Only NumPy and Pure Math

Welcome to **MiniML-Toolkit**, a DIY machine learning sandbox where we rebuild models from scratch using nothing but **NumPy** and a love for math.

No scikit-learn. No black boxes. Just gradients, dot products, and clean Python code.

---

## 📦 Features

✅ Simple Linear Regression (1 feature)  
✅ Multiple Linear Regression (n features)  
✅ Gradient Descent optimizer  
✅ Custom Standard Scaler  
✅ Manual Train/Test Split  
✅ R², MSE, RMSE Metrics — from scratch  

---

## 🧠 Project Structure

```

MiniML-Toolkit/
├── models/
│   ├── simple_linear_regression.py      # SLR model (1 X, 1 y)
│   └── multiple_linear_regression.py    # MLR model (n X, 1 y)
│   └── and more !!!
├── utils/
│   ├── standard_scaler.py               # Custom StandardScaler class
│   └── model_utils.py                    # Normalizer and Train Test Splitter
├── app.py                              # Run MLR/SLR training + evaluation
├── requirements.txt
└── README.md

````

---

## 🛠️ Installation

### 📌 Using `venv` (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

### 📌 Or use `conda`

```bash
conda create -n miniml python=<any version> -y
conda activate miniml
pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python app.py
```

modify `app.py` directly to experiment with parameters, datasets, and iterations.

---

## 📈 Sample Output

```text
[MLR] Initial MSE: 22.14
[MLR] Final RMSE: 4.11 | R²: 0.704

[SLR] Initial MSE: 33.8
[SLR] Final RMSE: 5.47 | R²: 0.52
```

---

## 💬 Why This Exists

Because nothing beats learning ML like writing it from scratch.
This toolkit is about understanding **how** things work, not just *that* they work.

More models (logistic regression, neural nets, etc.) coming soon!

---

## 🧪 Requirements

```bash
numpy
pandas
loguru

# optionally add matplotlib , seaborn for visuals
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Ideas? Fixes? New models?
Open an issue or PR — this project is open to all math-loving developers.

---

## 📜 License

MIT. Use it, learn from it, and feel free to build your own toolkit on top!
