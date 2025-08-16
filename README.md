# Logistic-Regression
Interactive Logistic Regression pipeline in Python — trains a binary classification model, evaluates accuracy, and predicts on new datasets with CSV output
# Logistic Regression Model – Python Implementation  

## 📌 Project Overview  
This project is a **custom-built Logistic Regression model pipeline** for binary classification tasks.  
It allows the user to:  
- Load a dataset (CSV) for training  
- Encode categorical features using a **custom masking function**  
- Train a Logistic Regression model using `statsmodels`  
- View model summary (coefficients, p-values, etc.)  
- Evaluate model accuracy  
- Use a separate CSV file for predictions  
- Save prediction results (probabilities + predicted classes) to a CSV  

---

## 🛠 Features  
- **Interactive Masking:** Map categorical variables into binary (0/1) values via user input.  
- **Model Training:** Fit a logistic regression model with `statsmodels`.  
- **Performance Metric:** Calculate accuracy on the training dataset.  
- **Predictions on New Data:** Take a prediction dataset, apply the same transformations, and output results.  
- **CSV Output:** Save results (with predicted probabilities and classes) to a CSV file.  

---

## ⚙ Tech Stack  
- **Python**  
- `pandas` – Data handling  
- `numpy` – Numerical operations  
- `statsmodels` – Logistic regression  
- `matplotlib` & `seaborn` – Visualization support  

---

## 📂 How to Use  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/YourUsername/your-repo-name.git
cd your-repo-name
```

### 2️⃣ Install Dependencies  
```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy
```

### 3️⃣ Run the Script  
```bash
python logistic_regression.py
```

### 4️⃣ Follow the Prompts  
- Enter your **training dataset file name**.  
- Provide the **target column** and **feature columns**.  
- Map categorical values via the interactive masking system.  
- The script trains the model and prints the **summary** + **accuracy**.  
- Enter the **prediction dataset file name**.  
- The script outputs predictions in a `.csv` file you name.  

---

## 📊 Example Output  
**Model Summary:**  
- Coefficients & p-values  
- Log-Likelihood & McFadden’s R²  

**Prediction CSV File:**  
| Feature1 | Feature2 | Predicted_Prob | Predicted_Class |  
|----------|----------|---------------|----------------|  
| ...      | ...      | 0.85          | 1              |  
| ...      | ...      | 0.32          | 0              |  

---

## 🧠 Why Logistic Regression?  
Logistic Regression is a fundamental supervised learning algorithm for binary classification problems.  
It’s widely used for tasks such as:  
- Spam detection  
- Medical diagnosis (disease vs. no disease)  
- Customer churn prediction  
- Admission/rejection predictions  

---

## 📜 License  
This project is open-source and available under the **MIT License**.  
