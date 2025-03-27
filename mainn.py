#!/usr/bin/env python
# coding: utf-8

# In[11]:


from flask import Flask, jsonify
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def health():
    try:
        # Load the dataset
        diabetes_path = "C:/ml1/diabetes.csv"  # Make sure this file path is correct
        Diabetes = pd.read_csv(diabetes_path)

        # Define features and target
        X = Diabetes[['Glucose', 'BMI', "Age"]].values
        y = Diabetes['Outcome'].values

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

        # Initialize classifiers
        clf1 = LinearDiscriminantAnalysis()
        clf2 = LogisticRegression()
        svc = SVC(probability=True)
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10], 'gamma': [0.1, 0.01]}
        clf3 = GridSearchCV(svc, parameters)
        clf4 = DecisionTreeClassifier(criterion="entropy", random_state=50, max_depth=4, min_samples_leaf=6)
        clf5 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf6 = AdaBoostClassifier(n_estimators=50)
        clf7 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=1, random_state=0, max_features='sqrt')

        # Create voting classifier
        eclf1 = VotingClassifier(estimators=[
            ('lda', clf1), 
            ('LogR', clf2), 
            ('svm', clf3), 
            ('dt', clf4), 
            ('rf', clf5), 
            ('Adaboost', clf6), 
            ('Graboost', clf7)], 
            voting='soft', verbose=True
        )
        
        # Fit the model
        eclf1.fit(X_train, y_train)

        # Predictions
        prediction = eclf1.predict(X_test)

        # Confusion Matrix
        cm = metrics.confusion_matrix(y_test, prediction, labels=eclf1.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=eclf1.classes_)
        disp.plot()
        plt.savefig('confusion_matrix.png')


        # ROC Curve
        RocCurveDisplay.from_estimator(eclf1, X_test, y_test)
        plt.savefig('roc_curve.png')


        # Scatter Plot
        y_score = eclf1.predict_proba(X_test)[:, 1]
        fig = px.scatter(
            x=X_test[:, 0], y=X_test[:, 1], color=y_score, color_continuous_scale='tropic',
            symbol=y_test, labels={
                'color': 'Probability Score', 
                'symbol': 'Ground Truth Label',
                'x': 'Glucose',  # Example feature
                'y': 'BMI'       # Example feature
            }
        )
        fig.update_traces(marker_size=12, marker_line_width=1.5)
        fig.update_layout(title='Scatter Plot of Diabetes diagnostics',legend_orientation='h')
        fig.show()
        fig.write_html("scatter_plot.html")


        # Return results
        return jsonify({
            "Training Accuracy": eclf1.score(X_train, y_train),
            "Testing Accuracy": eclf1.score(X_test, y_test)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




