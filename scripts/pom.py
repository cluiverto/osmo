import deepchem as dc
from rdkit import Chem
import pandas as pd
import numpy as np
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.utils.data_utils import get_class_imbalance_ratio, IterativeStratifiedSplitter
from openpom.models.mpnn_pom import MPNNPOMModel
from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde




def pom_plot(model, dataset, is_preds=False):
   pom_embeds = model.predict_embedding(dataset)
   required_desc = list(dataset.tasks)


   # Define type dictionaries for odor categories to be visualized
   type1 = {'floral': '#F3F1F7', 'subs': {'muguet': '#FAD7E6', 'lavender': '#8883BE', 'jasmin': '#BD81B7'}}
   type2 = {'meaty': '#F5EBE8', 'subs': {'savory': '#FBB360', 'beefy': '#7B382A', 'roasted': '#F7A69E'}}
   type3 = {'ethereal': '#F2F6EC', 'subs': {'cognac': '#BCE2D2', 'fermented': '#79944F', 'alcoholic': '#C2DA8F'}}
  
   # Perform Principal Component Analysis (PCA) to reduce the dimensionality of the embeddings to 2 components
   pca = PCA(n_components=2, iterated_power=10)
   reduced_features = pca.fit_transform(pom_embeds)


   # Get the variance explained by the first two principal components
   variance_explained = pca.explained_variance_ratio_
   variance_pc1 = variance_explained[0]
   variance_pc2 = variance_explained[1]


   # If is_preds is True, use the model to make predictions on the dataset
   if is_preds:
       y_preds = model.predict(dataset)
       # Set a threshold for predictions
       threshold = np.percentile(y_preds, 95, axis=0)
       y = (y_preds >= threshold).astype(int)
   else:
       # Otherwise, use the true labels from the dataset
       y = dataset.y


   # Define a grid of points for Kernel Density Estimation (KDE)
   x_grid, y_grid = np.meshgrid(np.linspace(reduced_features[:, 0].min(), reduced_features[:, 0].max(), 500),
                                np.linspace(reduced_features[:, 1].min(), reduced_features[:, 1].max(), 500))
   grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()])


   # Define a function to get KDE values for a specific label
   def get_kde_values(label):
       plot_idx = required_desc.index(label)
       label_indices = np.where(y[:, plot_idx] == 1)[0]
       kde_label = gaussian_kde(reduced_features[label_indices].T)
       kde_values_label = kde_label(grid_points)
       kde_values_label = kde_values_label.reshape(x_grid.shape)
       return kde_values_label
  
   # Define a function to plot contours for a given type dictionary
   def plot_contours(type_dictionary, bbox_to_anchor):
       main_label = list(type_dictionary.keys())[0]
       plt.contourf(x_grid, y_grid, get_kde_values(main_label), levels=1, colors=['#00000000',type_dictionary[main_label],type_dictionary[main_label]])
       legend_elements = []
       for label, color in type_dictionary['subs'].items():
           plt.contour(x_grid, y_grid, get_kde_values(label), levels=1, colors=color, linewidths=2)
           legend_elements.append(Patch(facecolor=color, label=label))
       legend = plt.legend(handles=legend_elements, title=main_label, bbox_to_anchor=bbox_to_anchor)
       legend.get_frame().set_facecolor(type_dictionary[main_label])
       plt.gca().add_artist(legend)


   # Create a figure and plot contours for different types
   plt.figure(figsize=(15, 10))
   plt.title('KDE Density Estimation with Contours in Reduced Space')
   plt.xlabel(f'Principal Component 1 ({round(variance_pc1*100, ndigits=2)}%)')
   plt.ylabel(f'Principal Component 2 ({round(variance_pc2*100, ndigits=2)}%)')
   plot_contours(type_dictionary=type1, bbox_to_anchor = (0.2, 0.8))
   plot_contours(type_dictionary=type2, bbox_to_anchor = (0.9, 0.4))
   plot_contours(type_dictionary=type3, bbox_to_anchor = (0.3, 0.1))


   # Display the plot
   plt.show()
   plt.close()