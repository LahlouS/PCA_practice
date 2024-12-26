import numpy as np
import pandas as pd

# Lets sum-up our continuous variables
def continuous_variable_summary(df, col_name):
	mean = []
	var = []
	std = []
	min_vals = []
	max_vals = []
	med = []

	for col in col_name:
		mean.append(df[col].mean())
		var.append(df[col].var())
		std.append(df[col].std())
		min_vals.append(df[col].min())
		max_vals.append(df[col].max())
		med.append(df[col].median())

	summary = pd.DataFrame({
		"mean": mean,
		"median": med,
		"variance": var,
		"std_dev": std,
		"min": min_vals,
		"max": max_vals
	}, index=col_name)

	return summary