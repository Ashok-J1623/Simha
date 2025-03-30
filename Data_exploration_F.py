#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_classif
from skrebate import MultiSURF
    
class DataExploration: 
    def __init__(self):
        self.data = None
        self.target = None
    
    # Method to load data from SQL or CSV
    def load_data(self, file_path=None, sql_query=None, connection=None):
        if file_path:
            self.data = pd.read_csv(file_path)
        elif sql_query and connection:
            self.data = pd.read_sql(sql_query, connection)
        else:
            print("Please provide either a file path for CSV or SQL query with connection.")
    
    # Method to view data structure, size, and edit options
    def data_info(self):
        if self.data is not None:
            print("Data Info:")
            print(self.data.info())
            print("\nData Size:", self.data.shape)
            print("\nData Preview:")
            print(self.data.head())
            
            # Edit options
            var = input("Enter variable name to edit or 'exit': ")
            while var != 'exit':
                if var in self.data.columns:
                    new_name = input(f"Enter new name for '{var}': ")
                    self.data.rename(columns={var: new_name}, inplace=True)
                var = input("Enter variable name to edit or 'exit': ")
        else:
            print("No data loaded.")
    
    # Method for descriptive statistics and visualization
    def data_statistics(self):
        if self.data is not None:
            print("Descriptive Statistics:\n")

            # General Descriptive Statistics
            print(self.data.describe(exclude=['object']))
            print("\n")

        # Distribution-Specific Statistics
            for col in self.data.select_dtypes(include=[np.number]).columns:
                print(f"Feature: {col}")
                print("-" * 40)

                # Normal Distribution
                mean = self.data[col].mean()
                median = self.data[col].median()
                std_dev = self.data[col].std()
                #print(f"Normal Distribution:")
                #print(f"  Mean: {mean:.2f}")
                #print(f"  Median: {median:.2f}")
                #print(f"  Standard Deviation: {std_dev:.2f}")

                # Uniform Distribution
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                #print(f"Uniform Distribution:")
               #print(f"  Min: {min_val}")
                #print(f"  Max: {max_val}")

                # Binomial Distribution (Approximation)
                num_trials = len(self.data[col].dropna())
                prob_success = (self.data[col] > 0).sum() / num_trials
                #print(f"Binomial Distribution (Approximated):")
                #print(f"  Number of Trials (n): {num_trials}")
                #print(f"  Probability of Success (p): {prob_success:.2f}")

                # Poisson Distribution (Approximation)
                rate_param = mean  # For Poisson, λ is approximated by the mean
                #print(f"Poisson Distribution:")
                #print(f"  Rate Parameter (λ): {rate_param:.2f}")

                # Exponential Distribution (Decay Rate)
                decay_rate = 1 / mean if mean != 0 else 0  # Avoid division by zero
                #print(f"Exponential Distribution:")
                #print(f"  Decay Rate (λ): {decay_rate:.2f}")

                # Skewness and Kurtosis
                skewness = self.data[col].skew()
                kurtosis = self.data[col].kurt()
                #print(f"Additional Metrics:")
                #print(f"  Skewness: {skewness:.2f}")
                #print(f"  Kurtosis: {kurtosis:.2f}")

                #print("\n")

            # Handle Missing Values
            missing_values = self.data.isnull().sum()
           # print("Missing Values:")
           # print(missing_values)

            # Visualization: Subplots for Distribution Plots
            num_features = len(self.data.select_dtypes(include=[np.number]).columns)
            rows = (num_features + 2) // 3  # 3 plots per row (adjust as needed)
            fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))  # Customize figure size
            axes = axes.flatten()  # Flatten to make indexing easier

            for i, col in enumerate(self.data.select_dtypes(include=[np.number]).columns):
                # Calculate statistics
                mean = self.data[col].mean()
                median = self.data[col].median()
                std_dev = self.data[col].std()
                min_val = self.data[col].min()
                max_val = self.data[col].max()

                # Plot density
                sns.kdeplot(self.data[col], fill=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("Density")

                # Overlay statistics
                axes[i].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
                axes[i].axvline(median, color='blue', linestyle='-.', label=f'Median: {median:.2f}')
                axes[i].axvline(min_val, color='green', linestyle='-', label=f'Min: {min_val:.2f}')
                axes[i].axvline(max_val, color='orange', linestyle='-', label=f'Max: {max_val:.2f}')
                axes[i].fill_betweenx(y=[0, axes[i].get_ylim()[1]], x1=mean - std_dev, x2=mean + std_dev, color='purple', alpha=0.2, label=f'Std Dev Range')

                # Add legend
                axes[i].legend(loc='upper right')

            # Remove unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()

            # Handle Missing Values
            missing_values = self.data.isnull().sum()
            print("Missing Values:")
            print(missing_values)

        else:
            print("No data loaded.")
            
    def visualize_categorical_features(self):
        if self.data is not None:
        # Select categorical columns
            categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
            for col in categorical_columns:
                print(f"{col}: {self.data[col].nunique()} unique values")

            filtered_columns = [col for col in categorical_columns if self.data[col].nunique() <= 5]
            print("Filtered Columns:", filtered_columns)
            filtered_columns = [col for col in filtered_columns if col != 'LoanID']
            dropped_columns = [col for col in categorical_columns if self.data[col].nunique() > 5] + ['LoanID']
            print(f"Filtered Columns (to visualize): {filtered_columns}")
            print(f"Dropped Columns (not visualized): {dropped_columns}")


        if len(filtered_columns) == 0:
            print("No suitable categorical features found in the dataset (all have more than 20 unique categories).")
            return
        for col in filtered_columns:
            print(f"Visualizing: {col}")
            self.data[col].value_counts().plot(kind='bar')
            plt.title(f"Category Counts for {col}")
            plt.show()

        # Create subplots for visualization
        num_features = len(categorical_columns)
        rows = (num_features + 1) // 2  # 2 plots per row
        fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))  # Adjust figure size
        axes = axes.flatten()  # Flatten to iterate easily
        
        for i, col in enumerate(categorical_columns):
            # Bar Chart
            sns.countplot(x=self.data[col], ax=axes[i], order=self.data[col].value_counts().index, palette="viridis")
            axes[i].set_title(f"Bar Chart of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
            axes[i].tick_params(axis='x', rotation=45)  # Rotate labels for better readability
            
            # Add counts above bars
            for p in axes[i].patches:
                axes[i].annotate(f'{int(p.get_height())}', 
                                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                                 ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                                 textcoords='offset points')
        
        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
        # Pie Charts
        for col in categorical_columns:
            data = self.data[col].value_counts()
            plt.figure(figsize=(8, 6))
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(data)))
            plt.title(f"Pie Chart of {col}")
            plt.show()
        else:
            print("No data loaded.")
    
    # Method to calculate classification feature scores
    def feature_score_classification(self, target_column):
        if self.data is not None and target_column in self.data.columns:
            self.target = target_column
            features = self.data.drop(columns=[target_column])
            scores = {}

            # Prepare data
            X = pd.get_dummies(features, drop_first=True)
            y = self.data[target_column]

            # Chi-Square Feature Scores
            chi2_scores = chi2(X, y)[0]
            scores['Chi2'] = dict(zip(X.columns, chi2_scores))

            # Gini Decrease (Random Forest)
            rf = RandomForestClassifier()
            rf.fit(X, y)
            gini_scores = rf.feature_importances_
            scores['Gini Decrease'] = dict(zip(X.columns, gini_scores))

            # Visualization
            self.visualize_feature_scores(scores)
        else:
            print("Ensure data is loaded and the target column exists.")
            
            # Method to calculate regression feature scores 
    def feature_score_regression(self, target_column):
        if self.data is not None and target_column in self.data.columns:
            self.target = target_column
            features = self.data.drop(columns=[target_column])
            scores = {}

            # Prepare data
            X = pd.get_dummies(features, drop_first=False).select_dtypes(include=[np.number])
            y = self.data[target_column]

            # Linear Regression Coefficients
            lr = LinearRegression()
            lr.fit(X, y)
            regression_scores = lr.coef_
            scores['Regression Coefficients'] = dict(zip(X.columns, regression_scores))

            # Visualization
            self.visualize_feature_scores(scores)
        else:
            print("Ensure data is loaded and the target column exists.")
            
           #shared visualization      
    def visualize_feature_scores(self, scores):
        for method, feature_scores in scores.items():
            plt.figure(figsize=(10, 6))
            plt.bar(feature_scores.keys(), feature_scores.values())
            plt.title(f"{method} Feature Scores")
            plt.xticks(rotation=90)
            plt.ylabel('Score')
            plt.show()  
        
    def sample_data(self, n=None, frac=None, random_state=None, replace=False):
        """
        Method to sample data from the dataset and prepare categorical columns.

        Parameters:
            n (int): Number of rows to sample. Mutually exclusive with `frac`.
            frac (float): Fraction of the dataset to sample (e.g., 0.1 for 10%). Mutually exclusive with `n`.
            random_state (int): Seed for random number generator (for reproducibility).
            replace (bool): Whether to replace the original dataset with the sampled data.

        Returns:
            pandas.DataFrame: Sampled and prepared data.
        """
        if self.data is not None:
            if n is not None and frac is not None:
                print("Please specify only one of `n` or `frac`, not both.")
                return None
            elif n is not None:
                print(f"Sampling {n} rows...")
                sampled_data = self.data.sample(n=n, random_state=random_state)
            elif frac is not None:
                print(f"Sampling {frac * 100:.2f}% of the dataset...")
                sampled_data = self.data.sample(frac=frac, random_state=random_state)
            else:
                print("Please specify either `n` or `frac` for sampling.")
                return None

            # Dynamically drop high-cardinality categorical columns (likely identifiers)
            high_cardinality_cols = [col for col in sampled_data.select_dtypes(include=['object', 'category']).columns
                                     if sampled_data[col].nunique() / len(sampled_data) > 0.9]
            sampled_data = sampled_data.drop(columns=high_cardinality_cols, errors='ignore')

            # Select categorical columns with ≤ 5 unique values
            categorical_columns = sampled_data.select_dtypes(include=['object', 'category']).columns
            filtered_columns = [col for col in categorical_columns if sampled_data[col].nunique() <= 5]

            # Optionally replace the original dataset
            if replace:
                self.data = sampled_data
                print("The original dataset has been replaced with the sampled data.")

            print(f"High Cardinality Columns Dropped: {high_cardinality_cols}")
            print(f"Filtered Columns for visualization or scoring: {filtered_columns}")
            return sampled_data[filtered_columns]
        else:
            print("No data loaded.")
            return None

