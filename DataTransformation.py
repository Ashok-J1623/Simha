#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.express as px
import plotly.graph_objects as go

class DataTransformation:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the DataTransformation superclass with a pandas DataFrame.

        :param data: pandas DataFrame to transform
        """
        self.data = data

    def sample_data(self, n_samples: int = 10, random_state: int = None):
        """
        Sample a subset of rows from the dataset.

        :param n_samples: Number of rows to sample
        :param random_state: Random seed for reproducibility
        :return: Sampled pandas DataFrame
        """
        sampled_data = self.data.sample(n=n_samples, random_state=random_state)
        self.visualize_data(sampled_data, title=f"Sampled {n_samples} Rows")
        return sampled_data

    def select_columns(self, columns: list):
        """
        Select specific columns from the dataset.

        :param columns: List of column names to select
        :return: Filtered pandas DataFrame with selected columns
        """
        selected_data = self.data[columns]
        self.visualize_data(selected_data, title=f"Selected Columns: {columns}")
        return selected_data

    def select_rows(self, condition: callable):
        """
        Select rows based on a given condition.

        :param condition: A callable (function or lambda) that returns a boolean mask
        :return: Filtered pandas DataFrame with rows meeting the condition
        """
        filtered_data = self.data[condition(self.data)]
        self.visualize_data(filtered_data, title="Filtered Rows Based on Condition")
        return filtered_data

    def merge_data(self, other_data: pd.DataFrame, on: str, how: str = 'inner'):
        """
        Merge the current DataFrame with another DataFrame.

        :param other_data: DataFrame to merge with
        :param on: Column name to merge on
        :param how: Type of merge ('inner', 'outer', 'left', 'right')
        :return: Merged pandas DataFrame
        """
        merged_data = pd.merge(self.data, other_data, on=on, how=how)
        self.visualize_data(merged_data, title=f"Merged Data ({how.capitalize()} Join on '{on}')")
        return merged_data

    def unique_rows(self):
        """
        Get unique rows from the dataset.

        :return: pandas DataFrame with unique rows
        """
        unique_data = self.data.drop_duplicates()
        self.visualize_data(unique_data, title="Unique Rows")
        return unique_data

    def aggregate_and_groupby(self, group_by_columns: list, agg_dict: dict):
        """
        Perform aggregation and grouping on the dataset.

        :param group_by_columns: List of columns to group by
        :param agg_dict: Dictionary specifying aggregation methods for each column
        :return: Aggregated and grouped pandas DataFrame
        """
        grouped_data = self.data.groupby(group_by_columns).agg(agg_dict).reset_index()
        self.visualize_data(grouped_data, title=f"Grouped Data by {group_by_columns}")
        return grouped_data

    def create_pivot_table(self, values: str, index: list, columns: list, aggfunc: str = 'sum'):
        """
        Create a pivot table.

        :param values: Column to aggregate
        :param index: List of index columns
        :param columns: List of column values to create pivot table
        :param aggfunc: Aggregation function (default is 'sum')
        :return: pandas Pivot Table as DataFrame
        """
        pivot_table = pd.pivot_table(self.data, values=values, index=index, columns=columns, aggfunc=aggfunc)
        self.visualize_data(pivot_table, title=f"Pivot Table ({aggfunc.capitalize()} Aggregation)")
        return pivot_table
    
    def discretize_variable(self, column: str, bins: int, strategy: str = 'uniform'):
        """
        Discretize a continuous variable into bins.

        :param column: Column to discretize
        :param bins: Number of bins
        :param strategy: Binning strategy ('uniform', 'quantile', or 'kmeans')
        :return: DataFrame with discretized column
        """
        discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
        self.data[column] = discretizer.fit_transform(self.data[[column]]).astype(int)
        return self.data

    def continuize_variable(self, columns: list):
        """
        Continuize discrete variables using one-hot encoding.

        :param columns: List of columns to continuize
        :return: DataFrame with one-hot encoded columns
        """
        encoder = OneHotEncoder(sparse=False, drop='first')
        encoded_data = encoder.fit_transform(self.data[columns])
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=encoder.get_feature_names_out(columns),
            index=self.data.index
        )
        self.data = pd.concat([self.data.drop(columns, axis=1), encoded_df], axis=1)
        return self.data

    def impute_missing_values(self, strategy: str = 'mean'):
        """
        Impute missing values in the dataset.

        :param strategy: Imputation strategy ('mean', 'median', 'most_frequent', or 'constant')
        :return: DataFrame with imputed values
        """
        imputer = SimpleImputer(strategy=strategy)
        self.data[:] = imputer.fit_transform(self.data)
        return self.data
    

    def select_relevant_features(self, target_column: str, k: int = 5):
        """
        Select the most relevant features using statistical tests.

        :param target_column: Column containing target labels
        :param k: Number of features to select
        :return: DataFrame with selected features
        """
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        selector = SelectKBest(score_func=f_classif, k=k)
        selected_features = selector.fit_transform(X, y)
        selected_columns = X.columns[selector.get_support()]
        self.data = pd.concat([self.data[selected_columns], self.data[[target_column]]], axis=1)
        return self.data

    def normalize_features(self, columns: list):
        """
        Normalize features using Min-Max scaling.

        :param columns: List of columns to normalize
        :return: DataFrame with normalized features
        """
        scaler = MinMaxScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])
        return self.data
    
    def transpose_column(self, column: str):
        """
        Transpose a specific column into a new DataFrame.

        :param column: Name of the column to transpose
        :return: Transposed pandas DataFrame
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset.")
        transposed_data = pd.DataFrame(self.data[column]).transpose()
        return transposed_data
    
    def melt_data(self, id_vars: list, value_vars: list = None, var_name: str = "Variable", value_name: str = "Value"):
        """
        Melt the DataFrame from wide to long format.

        :param id_vars: Columns to keep as identifier variables
        :param value_vars: Columns to melt (default is all columns not in id_vars)
        :param var_name: Name for the variable column in the melted DataFrame
        :param value_name: Name for the value column in the melted DataFrame
        :return: Melted pandas DataFrame
        """
        melted_data = pd.melt(self.data, id_vars=id_vars, value_vars=value_vars, 
                              var_name=var_name, value_name=value_name)
        return melted_data


    def visualize_data(self, data: pd.DataFrame, title: str, viz_type: str = "scatter", **kwargs):
        """
        Visualize the data using Plotly with multiple visualization options.

        :param data: pandas DataFrame to visualize
        :param title: Title for the visualization
        :param viz_type: Type of visualization ('scatter', 'bar', 'box', 'line', 'pie', 'hist_kde')
        :param kwargs: Additional arguments for the plot (e.g., x, y, color)
        """
        fig = None

        if viz_type == "scatter":
            fig = px.scatter(data, title=title, **kwargs)
        elif viz_type == "bar":
            fig = px.bar(data, title=title, **kwargs)
        elif viz_type == "box":
            fig = px.box(data, title=title, **kwargs)
        elif viz_type == "line":
            fig = px.line(data, title=title, **kwargs)
        elif viz_type == "pie":
            fig = px.pie(data, title=title, **kwargs)
        elif viz_type == "hist_kde":
            if "x" in kwargs:
                fig = px.histogram(data, x=kwargs["x"], marginal="box", opacity=0.7, title=title)
                fig.update_traces(marker=dict(line=dict(width=1)))
            else:
                raise ValueError("For 'hist_kde', the 'x' parameter must be provided.")
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")

        if fig is not None:
            fig.show()



# # Example usage:
# if __name__ == "__main__":
#     # Sample DataFrame
#     data = pd.DataFrame({
#         'A': [1.5, 2.3, 3.1, np.nan, 4.8],
#         'B': [10, 20, 30, 40, 50],
#         'C': ['X', 'Y', 'Z', 'X', 'Y'],
#         'D': [0, 1, 0, 1, 0],
#         'Target': [1, 0, 1, 0, 1]
#     })
# 
#     transformer = DataTransformation(data)
# 
#     # Discretize column 'A'
#     discretized_data = transformer.discretize_variable(column='A', bins=3)
#     print("Discretized Data:")
#     print(discretized_data)
# 
#     # Continuize column 'C'
#     continuized_data = transformer.continuize_variable(columns=['C'])
#     print("\nContinuized Data:")
#     print(continuized_data)
# 
#     # Impute missing values
#     imputed_data = transformer.impute_missing_values(strategy='mean')
#     print("\nImputed Data:")
#     print(imputed_data)
# 
#     # Select relevant features
#     selected_features = transformer.select_relevant_features(target_column='Target', k=2)
#     print("\nSelected Relevant Features:")
#     print(selected_features)
# 
#     # Normalize columns 'B' and 'A'
#     normalized_data = transformer.normalize_features(columns=['B', 'A'])
#     print("\nNormalized Data:")
#     print(normalized_data)

# # Example usage:
# if __name__ == "__main__":
#     # Sample DataFrame
#     data = pd.DataFrame({
#         'A': [1, 2, 3, 4, 5],
#         'B': [10, 20, 30, 40, 50],
#         'C': [5, 4, 3, 2, 1],
#         'D': [1, 0, 1, 0, 1]
#     })
# 
#     transformer = DataTransformation(data)
# 
#     # Sampling data and visualizing
#     sampled_data = transformer.sample_data(n_samples=3)
# 
#     # Selecting columns and visualizing
#     selected_data = transformer.select_columns(['A', 'B'])
# 
#     # Getting unique rows and visualizing
#     unique_data = transformer.unique_rows()
# 

# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# 
# class DataTransformation:
#     def __init__(self, data: pd.DataFrame):
#         """
#         Initialize the DataTransformation superclass with a pandas DataFrame.
# 
#         :param data: pandas DataFrame to transform
#         """
#         self.data = data
# 
#     def visualize_data(self, data: pd.DataFrame, title: str, viz_type: str = "scatter", **kwargs):
#         """
#         Visualize the data using Plotly with multiple visualization options.
# 
#         :param data: pandas DataFrame to visualize
#         :param title: Title for the visualization
#         :param viz_type: Type of visualization ('scatter', 'bar', 'box', 'line', 'pie', 'hist_kde')
#         :param kwargs: Additional arguments for the plot (e.g., x, y, color)
#         """
#         fig = None
# 
#         if viz_type == "scatter":
#             fig = px.scatter(data, title=title, **kwargs)
#         elif viz_type == "bar":
#             fig = px.bar(data, title=title, **kwargs)
#         elif viz_type == "box":
#             fig = px.box(data, title=title, **kwargs)
#         elif viz_type == "line":
#             fig = px.line(data, title=title, **kwargs)
#         elif viz_type == "pie":
#             fig = px.pie(data, title=title, **kwargs)
#         elif viz_type == "hist_kde":
#             if "x" in kwargs:
#                 fig = px.histogram(data, x=kwargs["x"], marginal="box", opacity=0.7, title=title)
#                 fig.update_traces(marker=dict(line=dict(width=1)))
#             else:
#                 raise ValueError("For 'hist_kde', the 'x' parameter must be provided.")
#         else:
#             raise ValueError(f"Unsupported visualization type: {viz_type}")
# 
#         if fig is not None:
#             fig.show()
# 
# # Example usage:
# if __name__ == "__main__":
#     # Sample DataFrame
#     data = pd.DataFrame({
#         'Category': ['A', 'B', 'A', 'B', 'C'],
#         'Value1': [10, 15, 10, 20, 25],
#         'Value2': [1, 3, 5, 7, 9]
#     })
# 
#     transformer = DataTransformation(data)
# 
#     # Scatter plot
#     transformer.visualize_data(data, title="Scatter Plot Example", viz_type="scatter", x="Value1", y="Value2", color="Category")
# 
#     # Bar chart
#     transformer.visualize_data(data, title="Bar Chart Example", viz_type="bar", x="Category", y="Value1")
# 
#     # Box plot
#     transformer.visualize_data(data, title="Box Plot Example", viz_type="box", x="Category", y="Value1")
# 
#     # Line plot
#     transformer.visualize_data(data, title="Line Plot Example", viz_type="line", x="Value2", y="Value1")
# 
#     # Pie chart
#     transformer.visualize_data(data, title="Pie Chart Example", viz_type="pie", names="Category", values="Value1")
# 
#     # Histogram with KDE
#     transformer.visualize_data(data, title="Histogram with KDE Example", viz_type="hist_kde", x="Value1")

# 
# # Example usage:
# if __name__ == "__main__":
#     # Sample DataFrame
#     data = pd.DataFrame({
#         'ID': [1, 2, 3],
#         'Name': ['Alice', 'Bob', 'Charlie'],
#         'Maths': [90, 80, 85],
#         'Science': [95, 85, 88]
#     })
# 
#     transformer = DataTransformation(data)
# 
#     # Melt data
#     melted_data = transformer.melt_data(id_vars=['ID', 'Name'], value_vars=['Maths', 'Science'])
#     print("Melted Data:")
#     print(melted_data)

# In[ ]:




