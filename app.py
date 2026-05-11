import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# LOAD DATA
# -----------------------------------

df = pd.read_csv("HR_comma_sep.csv")

X = df.drop('left', axis=1)
y = df['left']

X = pd.get_dummies(X, drop_first=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = pickle.load(open("model.pkl", "rb"))

# -----------------------------------
# PAGE TITLE
# -----------------------------------

st.title("HR Attrition Prediction")

# -----------------------------------
# INPUTS
# -----------------------------------

satisfaction = st.slider(
    "Satisfaction Level",
    0.0,
    1.0,
    0.5
)

evaluation = st.slider(
    "Last Evaluation",
    0.0,
    1.0,
    0.5
)

projects = st.number_input(
    "Number of Projects",
    1,
    10,
    3
)

hours = st.number_input(
    "Monthly Hours",
    50,
    400,
    160
)

years = st.number_input(
    "Years in Company",
    1,
    20,
    3
)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("Predict"):

    sample = np.array([[
        satisfaction,
        evaluation,
        projects,
        hours,
        years,
        0,
        0,
        0,0,0,0,0,0,0,0,0,
        1,0
    ]])

    prediction = model.predict(sample)

    probability = model.predict_proba(sample)

    # -----------------------------------
    # RESULT
    # -----------------------------------

    if prediction[0] == 1:
        st.error("Employee Will Leave")
    else:
        st.success("Employee Will Stay")

    st.write(
        "Probability of Leaving:",
        round(probability[0][1] * 100,2),
        "%"
    )

# -----------------------------------
# MODEL ACCURACY
# -----------------------------------

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)

st.write("Model Accuracy:",
         round(accuracy * 100,2),
         "%")

# ===================================
# GRAPH 1
# SCATTER PLOT
# ===================================

fig1, ax1 = plt.subplots()

ax1.scatter(
    df['satisfaction_level'],
    df['left']
)

ax1.set_xlabel("Satisfaction Level")
ax1.set_ylabel("Left")

ax1.set_title("Scatter Plot")

st.pyplot(fig1)

# ===================================
# GRAPH 2
# LOGISTIC CURVE
# ===================================

graph_model = LogisticRegression()

X_graph = df[['satisfaction_level']]
y_graph = df['left']

graph_model.fit(X_graph, y_graph)

x_range = np.linspace(
    df.satisfaction_level.min(),
    df.satisfaction_level.max(),
    300
).reshape(-1,1)

y_curve = graph_model.predict_proba(
    x_range
)[:,1]

fig2, ax2 = plt.subplots()

ax2.plot(
    x_range,
    y_curve
)

ax2.set_xlabel("Satisfaction Level")
ax2.set_ylabel("Probability")

ax2.set_title("Logistic Regression Curve")

st.pyplot(fig2) 