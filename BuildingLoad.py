#!/usr/bin/env python
# coding: utf-8

# In[21]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
##from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class DataPipeline:
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def load_data(self, path):
        # Load your dataset
        data = pd.read_csv("C:/Users/lenovo/HVAC.csv")
        self.X = data.drop('Building Load (RT)', axis=1)  # Features
        self.y = data['Building Load (RT)']               # Target variable
        return self

    def preprocess_data(self):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Standardize the features using a scaler
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        return self

    def train_model(self):
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self):
        # Make predictions and evaluate accuracy
        y_pred = self.model.predict(self.X_test)
        importances = self.model.feature_importances_
        #accuracy = accuracy_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f"Returning: mse={mse}, r2={r2}")
       ## return f'Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}'
        return mse, r2
    
        ##return f'Model accuracy: {mse:.2f},{r2:.2f}'
       

    def visualize_feature_importance(self):
    # Ensure the model has the attribute `feature_importances_`
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
            feature_names = self.X.columns

        # Create the bar plot
            plt.figure(figsize=(10, 6))
            plt.bar(feature_names, feature_importances)
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            #plt.show()
            st.pyplot(plt)
            
        else:
            print("Visualization not available: the model does not support feature importance.")
            
def main():
    st.title("Interactive Machine Learning with DataPipeline")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Preview of your data:")
        st.write(data.head())
        
        # Ensure target variable exists
        if 'Building Load (RT)' not in data.columns:
            st.error("The target variable 'Building Load (RT)' is missing from the dataset!")
        else:
            # Initialize pipeline
            pipeline = DataPipeline(RandomForestRegressor())
            pipeline.load_data(data).preprocess_data().train_model()
            
            # Evaluate model
            mse, r2 = pipeline.evaluate_model()
            print(f"MSE: {mse}, R2: {r2}")
            st.success(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR-squared: {r2:.2f}")
            
            # Visualization
            st.subheader("Prediction Visualization")
            pipeline.visualize_feature_importance()
          

if __name__ == "__main__":
    main()

# Example usage
#pipeline = DataPipeline(RandomForestRegressor())
#pipeline.load_data('data.csv').preprocess_data().train_model()
#pipeline.visualize_feature_importance()
#print(pipeline.evaluate_model())



# Explanation:
# Class Definition (DataPipeline): The pipeline organizes the entire workflow into reusable steps (methods).
# 
# Methods:
# 
# load_data: Loads the dataset.
# 
# preprocess_data: Splits and preprocesses the data (e.g., scaling features).
# 
# train_model: Trains the specified model.
# 
# evaluate_model: Tests the model and evaluates its performance.
# 
# Encapsulation: By using a class, the logic is modular, easy to reuse, and adaptable to different datasets or models.

# In[ ]:




