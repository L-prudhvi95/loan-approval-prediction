# 🏦 Loan Approval Prediction Using Machine Learning

## 📌 Project Overview

The **Loan Approval Prediction System** is a machine learning-based application that predicts whether a loan application is likely to be approved or rejected based on the applicant's financial and personal information.

The project uses Python and Scikit-learn to preprocess applicant data, train classification models, and generate predictions. A user-friendly interface built with Streamlit allows users to enter applicant details and view the prediction.

The main objective is to demonstrate how machine learning can support loan application analysis by identifying patterns in historical loan data.

> **Disclaimer:** This project is developed for educational purposes. Predictions are estimates from a trained model and must not be used as the sole basis for real-world lending decisions.

## 🎯 Project Objectives

- Analyze historical loan application data.
- Handle missing values and preprocess numerical and categorical features.
- Train and evaluate machine learning classification models.
- Compare Logistic Regression and Random Forest.
- Predict loan approval status using applicant information.
- Build an interactive web application using Streamlit.
- Deploy the project online and make it accessible to users.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Preprocessing, model training and evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Matplotlib / Seaborn | Data visualization, if used |
| Git & GitHub | Version control and project hosting |

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression
A supervised classification algorithm used as a baseline model for predicting loan approval.

### 2. Random Forest Classifier
An ensemble learning algorithm that combines predictions from multiple decision trees to classify loan applications.

The models are evaluated using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix. The final model should be selected based on actual evaluation results, not accuracy alone.

## 📊 Dataset

The project uses a loan prediction dataset containing applicant information and historical loan approval outcomes.

**Suggested dataset source:** [Search for Loan Prediction Problem datasets on Kaggle](https://www.kaggle.com/datasets?search=loan+prediction+problem)

The commonly used dataset includes the following columns:

| Feature | Description |
|---|---|
| Loan_ID | Unique loan application identifier |
| Gender | Applicant's gender |
| Married | Marital status |
| Dependents | Number of dependents |
| Education | Education status |
| Self_Employed | Employment category |
| ApplicantIncome | Applicant's income |
| CoapplicantIncome | Co-applicant's income |
| LoanAmount | Requested loan amount |
| Loan_Amount_Term | Loan repayment term |
| Credit_History | Credit history indicator |
| Property_Area | Urban, semi-urban or rural |
| Loan_Status | Historical approval outcome (target) |

**Target variable:** `Loan_Status`

- `Y` — Loan approved
- `N` — Loan not approved

The dataset should be downloaded from Kaggle and placed in the project's `data/` directory as `train.csv`. Confirm that its column names match the code before training.

## ⚙️ Project Workflow

1. **Data collection:** Obtain the historical loan dataset.
2. **Data exploration:** Examine columns, distributions and missing values.
3. **Data preprocessing:** Handle missing values and encode categorical variables.
4. **Feature preparation:** Separate input features from the target variable.
5. **Model training:** Train Logistic Regression and Random Forest classifiers.
6. **Model evaluation:** Compare performance on a held-out test set.
7. **Model persistence:** Save the trained preprocessing and prediction pipeline.
8. **Application development:** Collect applicant details through the Streamlit interface.
9. **Prediction:** Display the predicted loan status and, where supported, an estimated approval probability.
10. **Deployment:** Publish the application online.

## 📁 Project Structure

```text
Loan_Approval_Prediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── loan_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── eda.py
│   ├── preprocessing.py
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Important:** `loan_model.pkl` is generated after successfully running the model training script. It does not exist automatically when you create the folders. Your final structure should match the files actually present in your project.

## 💻 Installation and Setup

### Prerequisites

Install the following:

- Python
- Visual Studio Code
- Git (optional, for GitHub upload)

### Step 1: Clone the repository

After uploading the project to GitHub, clone it using:

```bash
git clone https://github.com/YOUR-USERNAME/Loan_Approval_Prediction.git
cd Loan_Approval_Prediction
```

Replace `YOUR-USERNAME` with your GitHub username.

If you are already working in your local project folder, open that folder directly in VS Code instead.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

For reproducible deployment, use tested and compatible dependency versions after confirming that the project runs successfully.

### Step 4: Add the dataset

Download the dataset from Kaggle and save the training file at:

```text
data/train.csv
```

### Step 5: Train the model

From the project's root directory, run:

```bash
python src/train_model.py
```

The script should preprocess the data, train and evaluate the models, and save the selected fitted pipeline to:

```text
models/loan_model.pkl
```

If training fails, resolve that error before launching the application.

### Step 6: Run the application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## 🖥️ Application Features

- Interactive form for applicant details.
- Input fields for income, loan amount, loan term and credit history.
- Selection fields for education, dependents, employment and property area.
- Machine learning-based approval prediction.
- Estimated approval probability, if supported by the trained model.
- User-friendly results and explanatory messages.

## 📈 Model Evaluation

The model should be evaluated on a separate test dataset. Record your actual results after running `train_model.py`.

| Evaluation metric | Actual result |
|---|---|
| Logistic Regression accuracy | Add measured result |
| Random Forest accuracy | Add measured result |
| Precision | Add measured result |
| Recall | Add measured result |
| F1-score | Add measured result |

Accuracy alone does not establish that a model is reliable. For loan-related classification, review false approvals, false rejections, class imbalance and the quality of the dataset as well.

## 🚀 Deployment

The application can be deployed using **Streamlit Community Cloud**.

1. Upload your project to a GitHub repository.
2. Ensure `app.py` and `requirements.txt` are included.
3. Ensure the trained model is available to the deployed application, or configure a reliable model-download/training process.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/).
5. Sign in with GitHub and create a new app.
6. Select your repository, branch and `app.py` as the main file.
7. Deploy the application and test it using sample inputs.

Follow the platform's current file-size and dependency requirements before uploading the trained model.

## 👥 Team Contributions

This project was developed as a collaborative academic project with four team members.

- **Member 1 — Data Collection and EDA:** Dataset research, data understanding and exploratory analysis.
- **Member 2 — Data Preprocessing:** Missing-value handling, categorical encoding and feature preparation.
- **Member 3 — Model Development:** Training classification models, evaluating performance and saving the selected model.
- **Member 4 — Application and Deployment:** Developing the user interface, integrating the trained model, testing and deployment.

Update these descriptions with the actual work completed by each team member.

## 🔮 Future Enhancements

- Compare additional classification algorithms.
- Improve validation and hyperparameter tuning.
- Add clearer explanations of prediction results.
- Improve input validation and error handling.
- Add model monitoring and retraining workflows.
- Improve accessibility and responsive UI design.
- Evaluate fairness, data quality and prediction reliability.

## 🎓 Learning Outcomes

Through this project, we explored:

- Supervised machine learning and classification.
- Data cleaning and preprocessing.
- Model training and performance evaluation.
- Saving and loading machine learning pipelines.
- Integrating a trained model with a web interface.
- Version control and application deployment.

## 👨‍💻 Author

**Name:** Your Name

**GitHub:** [Your GitHub Profile](https://github.com/L-prudhvi95)

**Project:** Loan Approval Prediction Using Machine Learning

## 📄 License

This project is intended for educational and academic purposes. Add an appropriate open-source license, such as the MIT License, if you want others to reuse and distribute the code.

---

⭐ If you find this project useful, consider starring the repository on GitHub.# loan-approval-prediction
