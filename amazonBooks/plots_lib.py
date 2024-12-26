import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from itertools import combinations

def plot_pca_scatter(data, 
					 vectors=None, 
					 vector_labels=None, 
					 contribution=None, 
					 data_label=None,
					 display_treshold=10,
					 figsize=(8, 6)):
	"""
	Plot a PCA scatter plot with optional vectors and sizes proportional to contributions.

	Parameters:
		data (ndarray): Array of shape (n_samples, 2) containing PCA components.
		vectors (ndarray, optional): Array of shape (n_vectors, 2) containing vectors to plot as arrows.
		vector_labels (list, optional): List of labels for the vectors.
		contribution (list or ndarray, optional): List or array of contributions for each datapoint to scale point sizes.
		data_label (list, optional): List of labels for each datapoint.
		figsize (tuple): Figure size.
	"""
	# Set the figure size
	plt.figure(figsize=figsize)

	# Handle contribution for scaling sizes
	if contribution is not None:
		contribution = np.array(contribution)
		sizes = 100 * (contribution / contribution.max())  # Normalize and scale contributions
	else:
		sizes = 50  # Default size for all points if contribution is not provided

	# Scatter plot of the PCA components
	plt.scatter(data[:, 0], data[:, 1], s=sizes, alpha=0.7, label='Data Points')

	# Set axis labels
	plt.xlabel('PC1')
	plt.ylabel('PC2')

	# Center the origin by setting x and y limits based on the data range
	data_limits = (data.min() - 1), (data.max() + 1)
	if vectors is not None:
		# Adjust limits to accommodate vectors
		vectors_limits = (vectors.min() - 1), (vectors.max() + 1)
		limits = (min(data_limits[0], vectors_limits[0]), max(data_limits[1], vectors_limits[1]))
	else:
		limits = data_limits

	plt.xlim(limits)
	plt.ylim(limits)

	# Add grid and origin lines for clarity
	plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
	plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
	plt.grid(True, linestyle=':', linewidth=0.5)

	# Plot vectors as arrows if provided
	if vectors is not None:
		for i, vector in enumerate(vectors):
			# Draw the arrow
			plt.arrow(0, 0, vector[0], vector[1], 
					  color='red', width=0.01, head_width=0.1, length_includes_head=True, alpha=0.8)
			# Add label if provided
			if vector_labels and i < len(vector_labels):
				plt.text(vector[0], vector[1], vector_labels[i], 
						 color='red', fontsize=10, ha='left', va='bottom')

	# Display labels for points with top 10% contributions
	if contribution is not None and data_label is not None:
		threshold = np.percentile(contribution, 100 - display_treshold)  # Top 10% threshold
		for i, (x, y, contrib) in enumerate(zip(data[:, 0], data[:, 1], contribution)):
			if contrib >= threshold:
				plt.text(x, y, data_label[i], fontsize=9, ha='center', va='center', color='blue')

	# Add title and legend
	plt.title('PCA Scatter Plot with Vectors and Contributions')
	if contribution is not None:
		plt.legend()

	# Show plot
	plt.show()




def plot_z(Z, factors=None, columns_name=["km", "price"]):
	plt.scatter(Z[:, 0], Z[:, 1])
	plt.xlabel(columns_name[0])
	plt.ylabel(columns_name[1])
	plt.title('PCA in the original basis (Z)')

	if factors is not None:
		# so it appears orthogonal
		plt.gca().set_aspect('equal', adjustable='box')
		scale = 2
		x1, y1 = factors[0, 0], factors[0, 1]
		x2, y2 = factors[1, 0], factors[1, 1]
		plt.plot([-x1*scale, x1*scale], [-y1*scale, y1*scale], color='blue', label='PC1')
		plt.plot([-x2*scale, x2*scale], [-y2*scale, y2*scale], color='red', label='PC2')
		plt.text(x1, y1, 'pc1', color='blue', ha='right')
		plt.text(x2, y2, 'pc2', color='red', ha='right')

	limits = (Z.min()), (Z.max())
	plt.xlim(limits)
	plt.ylim(limits)
	plt.show()
	
	
def plot_variables(data_point, vars_name=['km', 'price'], figsize=(8, 8)):
	"""
	Plot variable correlations with factorial axes.
	
	Parameters:
		data_point (numpy.ndarray): Array of data points for the variables.
		vars_name (list): List of variable names.
		figsize (tuple): Size of the figure in inches (width, height).
	"""
	dtps = []
	for i in range(len(vars_name)):
		x, y = data_point[i, 0], data_point[i, 1]
		dtps.append((x, y))
	
	# Set the figure size
	plt.figure(figsize=figsize)
	
	# Plot arrows instead of lines
	for i in range(len(dtps)):
		plt.arrow(0, 0, dtps[i][0], dtps[i][1], 
				  head_width=0.05, head_length=0.1, 
				  fc='grey', ec='grey', length_includes_head=True)
		plt.text(dtps[i][0], dtps[i][1], vars_name[i], color='grey')

	# Adjust plot limits and aspect ratio for clear vector visualization
	max_range = 0
	for coord in dtps:
		max_range = max(max_range, max(abs(coord[0]), abs(coord[1])))
	plt.xlim(-max_range, max_range)
	plt.ylim(-max_range, max_range)
	plt.axhline(0, color='blue', lw=0.5)
	plt.axvline(0, color='red', lw=0.5)
	
	plt.grid(color='gray', linestyle='--', linewidth=0.5)
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Variable Correlations with Factorial Axes")
	plt.show()


def display_pairplot(dataframe, hue=None):
	"""
	Display a pair plot of every variable pair in the dataframe.

	Parameters:
	dataframe (pd.DataFrame): The DataFrame containing the data.
	hue (str, optional): The column name to use for color encoding. Defaults to None.
	"""
	if not isinstance(dataframe, pd.DataFrame):
		raise ValueError("The input must be a pandas DataFrame.")

	# Generate the pair plot
	sns.pairplot(dataframe, hue=hue, diag_kind="kde")
	
	# Show the plot
	plt.show()

def display_pairplot_with_arrows(correlation_df):
	"""
	Display a pair plot of factorial axes combinations with arrows representing variable correlations.

	Parameters:
	correlation_df (pd.DataFrame): DataFrame with columns as factorial axes and rows as variables.
								   Each cell contains the correlation of a variable with a factorial axis.
								   Columns = ['FactorialAxis0', 'FactorialAxis1', ..., 'FactorialAxisp']
								   Index = [Variable names]
	"""
	if not isinstance(correlation_df, pd.DataFrame):
		raise ValueError("The input must be a pandas DataFrame.")
	
	# Extract factorial axes names
	factorial_axes = correlation_df.columns
	variable_names = correlation_df.index
	
	# Generate all combinations of factorial axes
	axis_combinations = list(combinations(factorial_axes, 2))
	
	# Create pair plot for each combination
	n_combinations = len(axis_combinations)
	n_cols = min(3, n_combinations)  # Number of plots per row
	n_rows = int(np.ceil(n_combinations / n_cols))
	
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
	axes = axes.flatten()  # Flatten axes to iterate easily

	for idx, (axis_x, axis_y) in enumerate(axis_combinations):
		ax = axes[idx]

		# Plot arrows for each variable
		for variable in variable_names:
			x = correlation_df.loc[variable, axis_x]
			y = correlation_df.loc[variable, axis_y]
			ax.arrow(0, 0, x, y, color='blue', alpha=0.8, head_width=0.03, length_includes_head=True)
			ax.text(x * 1.1, y * 1.1, variable, fontsize=9, color='black', ha='center', va='center')
		
		# Add grid and axis labels
		ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)
		ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
		ax.set_xlim(-1.1, 1.1)
		ax.set_ylim(-1.1, 1.1)
		ax.set_xlabel(axis_x)
		ax.set_ylabel(axis_y)
		ax.set_title(f"{axis_x} vs {axis_y}")
		ax.grid(alpha=0.3)

	# Hide unused subplots (if any)
	for idx in range(len(axis_combinations), len(axes)):
		axes[idx].axis("off")
	
	plt.tight_layout()
	plt.show()

def plot_cumulative_proportion(data):
	"""
	Plots the cumulative proportion of the elements in the given array.

	Parameters:
		data (array-like): Input array of shape (n,). The sum of the array should be equal to n.
	"""
	# Ensure the data is a NumPy array
	data = np.insert(np.array(data), 0, 0)
	# Compute the cumulative sum
	cumulative_sum = np.cumsum(data)

	# Compute the cumulative proportion
	total = np.sum(data)
	cumulative_proportion = cumulative_sum / total

	# Create the plot
	plt.figure(figsize=(8, 5))
	plt.plot(cumulative_proportion, marker='o', linestyle='-', color='b', label='Cumulative Proportion')
	plt.title('Cumulative Proportion Plot')
	plt.xlabel('Index')
	plt.ylabel('Cumulative Proportion')
	plt.ylim(0, 1.05)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.legend()
	plt.show()
