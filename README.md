\# SLTB vs Private Bus Preference Prediction



\## Project Overview



This project develops a machine learning system to predict whether a passenger is more likely to prefer an SLTB bus or a Private bus in Sri Lanka.



The project uses passenger travel and service-related information to train and evaluate several machine learning classification models.



\## Objectives



\- Predict passenger bus preference.

\- Compare different machine learning algorithms.

\- Evaluate model performance using classification metrics.

\- Develop a working prediction application using Streamlit.



\## Dataset



The dataset contains passenger-related information such as:



\- Gender

\- Age Group

\- Occupation

\- Monthly Income

\- Bus Usage Frequency

\- Bus Type Used

\- Trip Distance

\- Travel Purpose

\- Wait Time

\- Service Frequency

\- Fare

\- Comfort

\- Crowding

\- Cleanliness

\- Safety

\- AC

\- Driver

\- Reliability



The target variable is `Bus\_Preference`.



\## Machine Learning Models



The following models were implemented and evaluated:



1\. Decision Tree

2\. Logistic Regression

3\. Random Forest

4\. XGBoost



A Genetic Algorithm based feature selection approach was also implemented with XGBoost.



\## Model Performance



| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |

|---|---:|---:|---:|---:|---:|

| Decision Tree | 78.44% | 78.46% | 78.44% | 78.45% | 78.23% |

| Logistic Regression | 83.03% | 83.04% | 83.03% | 82.95% | 91.85% |

| Random Forest | 83.94% | 83.94% | 83.94% | 83.90% | 92.37% |

| XGBoost | 85.78% | 85.83% | 85.78% | 85.71% | 92.22% |



XGBoost achieved the highest accuracy among the evaluated baseline models.



\## Working Model



A Streamlit application is included in `app.py`.



The application allows users to enter passenger information and receive:



\- Predicted bus preference

\- Probability for each bus type



\## Project Structure



```text

SLTB-vs-Private-Bus-Preference/

│

├── architecture/

├── data/

├── models/

├── notebooks/

├── results/

├── src/

│

├── app.py

├── README.md

├── requirements.txt

└── .gitignore

