# HR Attrition Prediction using Machine Learning

Overview

This project is a Machine Learning based web application developed using Streamlit that predicts whether an employee is likely to leave a company or stay.

The application uses employee-related information such as:

* Satisfaction level
* Last evaluation score
* Number of projects
* Monthly working hours
* Years at the company

Based on these inputs, the trained Logistic Regression model predicts employee attrition and displays:

* Prediction result
* Probability of leaving
* Model accuracy
* Visualization graphs

Features

* Interactive user interface using Streamlit
* Employee attrition prediction
* Probability score generation
* Model accuracy display
* Scatter plot visualization
* Logistic Regression probability curve

Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Pickle

Dataset

Dataset used:
`HR_comma_sep.csv`

The dataset contains employee details and attrition status.

Target Column:

* `left`

  * `1` → Employee Left
  * `0` → Employee Stayed


Machine Learning Algorithm

Algorithm Used:

* Logistic Regression

The model is trained and saved as:
`model.pkl`

Project Structure

bash
HR-Attrition-Prediction/
│
├── app.py
├── model.pkl
├── HR_comma_sep.csv
├── requirements.txt
└── README.md


Installation

1. Clone Repository

bash
git clone https://github.com/your-username/HR-Attrition-Prediction.git

2. Navigate to Project Folder

bash
cd HR-Attrition-Prediction

3. Install Dependencies

bash
pip install -r requirements.txt

Run the Application

bash
streamlit run app.py


After running the command, the application opens in your browser automatically.

Input Parameters

The user provides:

* Satisfaction Level
* Last Evaluation
* Number of Projects
* Monthly Hours
* Years in Company

Output

The application predicts:

* Employee Will Stay
  OR
* Employee Will Leave

It also displays:

* Probability of Leaving
* Model Accuracy

Visualizations

1. Scatter Plot

Shows relationship between:

* Satisfaction Level
* Employee Attrition

2. Logistic Regression Curve

Displays probability trend of employee attrition based on satisfaction level.


Example Prediction

| Satisfaction | Evaluation | Projects | Hours | Years | Prediction          |
| ------------ | ---------- | -------- | ----- | ----- | ------------------- |
| 0.45         | 0.80       | 4        | 210   | 3     | Employee Will Leave |


Future Improvements

* Add more ML algorithms
* Improve UI design
* Deploy using Streamlit Cloud
* Add real-time analytics dashboard
* Feature importance visualization

Conclusion

This project demonstrates how Machine Learning can be used to predict employee attrition effectively using Logistic Regression. By analyzing employee-related factors and visualizing the results, the application helps understand employee behavior and supports better decision-making in HR management.
