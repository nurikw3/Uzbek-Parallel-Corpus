![image](/assets/logo1.png)

# Southern Uzbek Machine Translation & Analysis

This repository contains a solution for the Machine Translation task focusing on **Southern Uzbek (Arabic Script)**, a low-resource language.

The project evaluates Zero-Shot performance of **NLLB-200** and **M2M100** on two tasks:
1. Translation: Southern Uzbek -> English.
2. Normalization: Southern Uzbek -> Standard Uzbek (Latin).

## 📂 Project Structure

```bash
/submission
│── prepare_data.py      # Downloads dataset, cleans, generates references, EDA
│── translate_model_1.py # Inference using NLLB-200 (Supports Arabic script)
│── translate_model_2.py # Inference using M2M100 (Baseline)
│── run_evaluation.py    # Calculates metrics (BLEU, chrF) and logs to MLflow
│── dataset.csv          # Processed dataset (Generated)
│── report.txt           # Final analytical report
└── README.md            
```

# Some EDA
![image](/assets/word_count_distribution.png)
![image](/assets/inter.jpeg)

# 📊 Results
![image](/assets/nllb.jpg)
![image](/assets/m2m.jpg)
# 🚀 Usage (via uv)
```bash
uv sync
uv run run_evaluation.py
mlflow ui --port 5001
```

> And open `http://localhost:5001` in your browser