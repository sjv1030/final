import pandas as pd
import numpy as np
import math
from scipy import stats

"""
@author: michelebradley
"""

class descriptive_stats(object):

    def __init__(self, df):
        self.df = df

    def get_stats(self, column_data):
        """calculates all of the stats to obtain uncertainty values
        requires input of desired confidence level
        returns a list of stats values for a column of a dataframe"""

        average = round(np.mean(column_data), 0)
        variance = round(np.var(column_data), 0)
        std_dev = round(np.std(column_data), 0)
        stats_list = [(average, variance, std_dev)]
        return stats_list

    def stats_df(self):
        """Generates a dataframe of the uncertainty stats
        Takes in dataframe where the second column starts the data we wish to summarize"""
        df = self.df

        labels = ["average", "variance", "std_dev"]
        uncertainty = pd.DataFrame({"average": ["average"], "variance": ["variance"], "std_dev": ["standard deviation"]})
        df = df.select_dtypes(include=['number'])
        number = len(df.columns)
        column_names = list(df)
        for i in range(1, number):
            column_data = df.iloc[:,i]
            column_name = column_names[i]
            values = self.get_stats(column_data)
            uncertainty_values = pd.DataFrame.from_records(data = values, columns = labels)
            uncertainty_values.index=[column_name]
            uncertainty = uncertainty.append(uncertainty_values)
        return uncertainty
