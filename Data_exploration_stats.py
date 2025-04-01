#!/usr/bin/env python
# coding: utf-8

# In[45]:


import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationMixin:
    @staticmethod
    def histogram(data1, data2=None,null_mean=None, title="Data Distribution", labels=None):
        """
         Visualize distributions for two-sample t-test with optional labels
         """
        plt.figure(figsize=(10, 6))
        sns.histplot(data1, bins=20, color='blue', label=labels[0] if labels else "Data 1", kde=True, alpha=0.6)
        if data2 is not None:
            sns.histplot(data2, bins=20, color='orange', label=labels[1] if labels else "Data 2", kde=True, alpha=0.6)
            plt.title(title)
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.show()
    @staticmethod
    def barplot(data, x, y, hue=None, title="Group-wise Means"):
        """
        Create a bar plot for group-wise mean values.
        Parameters:
        - data: DataFrame containing the data.
        - x: Column name for the categorical variable (e.g., groups).
        - y: Column name for the numerical variable (e.g., values to aggregate).
        - hue: Optional column for sub-groups within x (e.g., blocking variable).
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(x=x, y=y, hue=hue, data=data, ci=None, palette="viridis")
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        if hue:
            plt.legend(title=hue)
        plt.show()    

    @staticmethod
    def boxplot(data, group_column, value_column, title="Group-wise Distribution"):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_column, y=value_column, data=data)
        plt.title(title)
        plt.xlabel(group_column)
        plt.ylabel(value_column)
        plt.show()

    @staticmethod
    def heatmap(data, group1, group2, value_column, title="Interaction Effects"):
        pivot_table = data.pivot_table(values=value_column, index=group1, columns=group2, aggfunc='mean')
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
        plt.title(title)
        plt.xlabel(group2)
        plt.ylabel(group1)
        plt.show()

class DataAnalysisBase:
     def load_data(self, file_path):
        """
        Default implementation to load data from a CSV file.
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Data successfully loaded from {file_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

class Statistics(DataAnalysisBase,VisualizationMixin):
    def __init__(self):
        self.data = None

    def load_data(self, file_path):
        """
        Method to load data from a CSV file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data successfully loaded from {file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None

    def hypothesis_test(self, sample_data, null_mean):
        """
        One-sample t-test
        """
        t_stat, p_value = ttest_1samp(sample_data, null_mean)
        return t_stat, p_value

    def two_sample_t_test(self, data1, data2):
        """
        Two-sample t-test
        """
        t_stat, p_value = ttest_ind(data1, data2)
        return t_stat, p_value

    def one_way_anova(self, group_column, value_column):
        """
        One-way ANOVA with descriptive statistics
        """
        model = ols(f'{value_column} ~ C({group_column})', data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        descriptive_stats = self.data.groupby(group_column)[value_column].describe()
        return anova_table, descriptive_stats

    def anova_blocking(self, block_column, group_column, value_column):
        """
        ANOVA with blocking (Randomized Block Design)
        """
        model = ols(f'{value_column} ~ C({block_column}) + C({group_column})', data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table

    def posthoc_test(self, group_column, value_column):
        """
        ANOVA Post-hoc Test (Pairwise Comparison)
        """
        posthoc_result = pairwise_tukeyhsd(endog=self.data[value_column], groups=self.data[group_column], alpha=0.05)
        return posthoc_result

    def two_way_anova(self, group1, group2, value_column):
        """
        Two-way ANOVA with interactions
        """
        model = ols(f'{value_column} ~ C({group1}) * C({group2})', data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table


# In[46]:


stat = Statistics()


# In[ ]:


stat.load_data("C:/users/lenovo/Telco_Customer_Churn.csv")
print(stat.data) 


# In[ ]:


print(stat.data.columns)


# In[ ]:


t_stat, p_value = stat.hypothesis_test(stat.data['MonthlyCharges'].dropna(), 70)
print(f"T-statistic: {t_stat}, P-value: {p_value}")


# In[ ]:


stat.data = stat.data.dropna(subset=['InternetService', 'MonthlyCharges'])  # Clean the data
anova_table, desc_stats = stat.one_way_anova(group_column='InternetService', value_column='MonthlyCharges')
print(anova_table)
print(desc_stats)


# In[ ]:





# In[19]:


stat.histogram(stat.data['MonthlyCharges'], null_mean=70, title="Monthly Charges Distribution")
stat.boxplot(stat.data, group_column='InternetService', value_column='MonthlyCharges', title="Internet Service vs Monthly Charges")


# In[43]:


# Prepare data for t-test
group1 = stat.data[stat.data['InternetService'] == 'Fiber optic']['MonthlyCharges'].dropna()
group2 = stat.data[stat.data['InternetService'] == 'DSL']['MonthlyCharges'].dropna()

# Perform t-test
t_stat, p_value = stat.two_sample_t_test(group1, group2)
print(f"Two-Sample t-Test Results: T-statistic = {t_stat}, P-value = {p_value}")

# Visualize distributions with labels
stat.histogram(group1, group2, title="Monthly Charges Distribution by Internet Service", labels=["Fiber optic", "DSL"])


# In[51]:


# Perform ANOVA with blocking
anova_table = stat.anova_blocking(block_column='PaymentMethod', group_column='Contract', value_column='MonthlyCharges')
print("ANOVA with Blocking Results:")
print(anova_table)

# Visualize means across groups
stat.barplot(stat.data, x='Contract', y='MonthlyCharges', hue='PaymentMethod', title="Monthly Charges by Contract and Payment Method")


# In[52]:


# Perform ANOVA Post-hoc Test
posthoc_result = stat.posthoc_test(group_column='InternetService', value_column='MonthlyCharges')

# Display results
print(posthoc_result)


# In[53]:


stat.boxplot(stat.data, group_column='InternetService', value_column='MonthlyCharges', title="Internet Service vs Monthly Charges")


# In[54]:


# Perform Two-Way ANOVA
anova_table = stat.two_way_anova(group1='InternetService', group2='Contract', value_column='MonthlyCharges')

# Display ANOVA table
print("Two-Way ANOVA Results:")
print(anova_table)


# In[55]:


stat.barplot(stat.data, x='Contract', y='MonthlyCharges', hue='PaymentMethod', title="Interaction Effects of Internet Service and Contract on Monthly Charges")


# In[57]:


stat.heatmap(stat.data, 'InternetService','Contract','MonthlyCharges', title="Interaction Effects")


# In[ ]:




