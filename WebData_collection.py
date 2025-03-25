#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#define class
class datacollection:
    #constructor method
    def __init__(self,url,api_key,lat,lon):
        self.url= url #initialize the data collection attribute
        self.api_key= api_key
        self.lat= lat
        self.lon= lon
    #display information
    def get_request(self,params=None): # Accept params as an optional argument

        if params is None: # Use the instance variables as default parameters

            params = {"api_key":self.api_key, "lat": self.lat,"lon": self.lon} # Use 'self.api_key' correctly

        response = requests.get(self.url, params=params) # Correct HTTP request

        response_json = response.json()  # Parse JSON directly
        return(response_json)
        print((list(response_json.keys())))

    # Function to create side-by-side plots
    def corr_plot_and_visualization(self, solar_data):
        # Extract data
        avg_dni = solar_data["outputs"]["avg_dni"]["monthly"]
        avg_ghi = solar_data["outputs"]["avg_ghi"]["monthly"]
        avg_lat_tilt = solar_data["outputs"]["avg_lat_tilt"]["monthly"]

        # Create DataFrame
        solar_df = pd.DataFrame({
            "Month": list(avg_dni.keys()),
            "Avg_DNI": list(avg_dni.values()),
            "Avg_GHI": list(avg_ghi.values()),
            "Avg_Lat_Tilt": list(avg_lat_tilt.values())
        })

        # Correlation Matrix
        corr_matrix = solar_df[["Avg_DNI", "Avg_GHI", "Avg_Lat_Tilt"]].corr()

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

        # First Plot: Line Graphs
        axes[0].plot(solar_df["Month"], solar_df["Avg_DNI"], marker='D', linestyle='-', color='blue', label='Avg_DNI')
        axes[0].plot(solar_df["Month"], solar_df["Avg_GHI"], marker='s', linestyle='--', color='orange', label='Avg_GHI')
        axes[0].plot(solar_df["Month"], solar_df["Avg_Lat_Tilt"], marker='o', linestyle='-.', color='brown', label='Avg_Lat_Tilt')
        for i, value in enumerate(solar_df["Avg_Lat_Tilt"]):
            axes[0].annotate(f"{value}", (solar_df["Month"][i], solar_df["Avg_Lat_Tilt"][i]),textcoords="offset points", xytext=(0, 5), 
                ha='center', fontsize=8)  
        axes[0].set_title('Monthly Solar Metrics And tilt optimization', fontsize=14)
        axes[0].set_xlabel('Month', fontsize=12)
        axes[0].set_ylabel('Irradiance (kWh/m²/day)', fontsize=12)
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Second Plot: Correlation Heatmap
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
        axes[1].set_title('Correlation Plot', fontsize=14)

        # Adjust layout and display
        plt.tight_layout()
        plt.savefig("C:/Users/lenovo/solar_plots.png")
        plt.show()

        # Optionally return the DataFrame
        return solar_df


# In[4]:


##solar=datacollection('https://developer.nrel.gov/api/solar/solar_resource/v1.json','KPI_key',40,-105)


# In[3]:


solar_data=solar.get_request()
solar.corr_plot_and_visualization(solar_data)


# In[ ]:




