import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_pca_scatter(data, vectors=None, vector_labels=None):

    # Scatter plot of the PCA components
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
    
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

    # Show plot
    plt.title('PCA Scatter Plot with Vectors and Labels')
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
	
    
def plot_variables(data_point, vars_name=['km', 'price']):
	x1, y1 = data_point[0, 0], data_point[0, 1]
	x2, y2 = data_point[1, 0], data_point[1, 1]
	
	# Plot arrows instead of lines
	plt.arrow(0, 0, x1, y1, head_width=0.05, head_length=0.1, fc='grey', ec='grey', length_includes_head=True)
	plt.arrow(0, 0, x2, y2, head_width=0.05, head_length=0.1, fc='grey', ec='grey', length_includes_head=True)

	# Add variable labels
	plt.text(x1, y1, vars_name[0], color='grey')
	plt.text(x2, y2, vars_name[1], color='grey')
	
	# Adjust plot limits and aspect ratio for clear vector visualization
	max_range = max(abs(x1), abs(y1), abs(x2), abs(y2)) + 0.1
	plt.xlim(-max_range, max_range)
	plt.ylim(-max_range, max_range)
	plt.axhline(0, color='blue', lw=0.5)
	plt.axvline(0, color='red', lw=0.5)
	#plt.gca().set_aspect('equal', adjustable='box')
	
	plt.grid(color='gray', linestyle='--', linewidth=0.5)
	plt.xlabel('PC1')
	plt.ylabel('PC2')
	plt.title("Variable Correlations with Factorial Axes")
	plt.show()