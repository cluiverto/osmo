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


TASKS = [
'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

featurizer = GraphFeaturizer()
input_file = 'curated_GS_LF_merged_4983.csv'

def make_dataset():
    smiles_field = 'curated_GS_LF_merged_4983nonStereoSMILES' # column that contains SMILES
    loader = dc.data.CSVLoader(tasks=TASKS,
    feature_field=smiles_field,
    featurizer=featurizer)
    dataset = loader.create_dataset(inputs=[input_file])
    return dataset


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




def cluster_plot(model, dataset, is_preds=False):
    pom_embeds = model.predict_embedding(dataset)
    required_desc = list(dataset.tasks)


    # Define type dictionaries for odor categories to be visualized
    type1 = {
        'sweet': '#F3F1F7',
            'subs': {
            'vanilla': '#F5DEB3',
            'caramellic': '#DEB887',
            'honey': '#FFD700',
            'sweet': '#FFE4B5',
            'buttery': '#FFEBCD',
            'creamy': '#FFF8DC',
            'milky': '#F0FFF0'
        }
    }
    type2 = {
        'woody': '#F2F6EC',
        'subs': {
            'cedar': '#BCE2D2',
            'sandalwood': '#79944F',
            'pine': '#C2DA8F',
            'vetiver': '#A9BA9D',
            'woody': '#8F9779'
        }
    }
    type3 = {
        'spicy': '#FFF0F5',
        'subs': {
            'cinnamon': '#CD5C5C',
            'clove': '#8B4513',
            'spicy': '#FF6347',
            'sharp': '#FA8072'
        }
    }

    # type5 = {
    #     'sweet': '#FFFACD',
    #     'subs': {
    #         'vanilla': '#F5DEB3',
    #         'caramellic': '#DEB887',
    #         'honey': '#FFD700',
    #         'sweet': '#FFE4B5',
    #         'buttery': '#FFEBCD',
    #         'creamy': '#FFF8DC',
    #         'milky': '#F0FFF0'
    #     }
    # }
    
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