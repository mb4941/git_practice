# Code to include week indicator in in 'hourly_utility_for_Julia.csv' and 'hourly_fuel_utility_for_Julia.csv'

from cmath import isnan
import pandas as pd
import numpy as np
import csv

# read in the data
data = pd.read_csv('/Users/miguelborrero/Desktop/Energy_Transitions/utility_fuel_hour_for_Julia.csv')
data2 = pd.read_csv('/Users/miguelborrero/Desktop/Energy_Transitions/utility_hour_for_Julia.csv')

# creating correspondance domain of hours in each week
hours_weekly = [[i*168 + j for j in np.arange(1, 169, 1)] for i in range(51)]
def get_week(hour):
    w = 0
    for index, vec in enumerate(hours_weekly):
        if hour in vec:
            w = index + 1
            break
    print(w)
    return w

data2['Week'] = data2.apply(lambda row: get_week(row['hour_of_year']) if not(np.isnan(row['hour_of_year'])) else 0, axis = 1)


data2.to_csv('/Users/miguelborrero/Desktop/Energy_Transitions/utility_hour_for_Julia.csv')


