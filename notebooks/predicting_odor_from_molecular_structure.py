import pyrfume
import pandas as pd
import plotly.express as px
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations
from ast import literal_eval  # Bezpieczniejsza alternatywa dla eval()
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # potrzebne do 3D


def load_and_merge_data():
    # Załaduj dane molekuł
    molecules = pyrfume.load_data('leffingwell/molecules.csv')
    # Załaduj dane behawioralne
    behavior = pyrfume.load_data('leffingwell/behavior_sparse.csv')

    # Połącz dane po indeksach
    merged_df = pd.merge(molecules, behavior, left_index=True, right_index=True)

    # Wybierz kolumny do dalszej analizy
    merged_df = merged_df.loc[:, ['IsomericSMILES', 'name', 'Labels']]

    # Usuń wiersze z brakującymi wartościami w kluczowych kolumnach
    merged_df = merged_df.dropna(subset=['IsomericSMILES', 'Labels'])

    return merged_df

def goodscents():
    a = pyrfume.load_data('goodscents/molecules.csv') 
    b = pyrfume.load_data('goodscents/behavior.csv')
    s = pyrfume.load_data('goodscents/stimuli.csv')
    df = pd.merge(b, s, left_index=True, right_index=True)
    full_df = pd.merge(df, a, on='CID', how='left') 
    return full_df



def plot_label_frequencies(df, labels_column='Labels', title='Częstość występowania etykiet zapachowych'):
    """
    Tworzy wykres bąbelkowy częstości występowania etykiet zapachowych.

    Parametry:
    -----------
    df : pd.DataFrame
        DataFrame zawierający kolumnę z etykietami.
    labels_column : str, opcjonalnie
        Nazwa kolumny zawierającej etykiety (domyślnie 'Labels').
        Etykiety mogą być listami lub stringami reprezentującymi listy.
    title : str, opcjonalnie
        Tytuł wykresu.

    Zwraca:
    --------
    fig : plotly.graph_objs._figure.Figure
        Obiekt wykresu Plotly.
    """

    # Jeśli etykiety są stringami, konwertujemy je na listy
    if df[labels_column].dtype == object and df[labels_column].apply(lambda x: isinstance(x, str)).all():
        df = df.copy()
        df[labels_column] = df[labels_column].apply(ast.literal_eval)

    # Rozpakuj listy etykiet do osobnych wierszy
    exploded = df.explode(labels_column)

    # Zlicz częstość występowania każdej etykiety
    label_counts = exploded[labels_column].value_counts().reset_index()
    label_counts.columns = ['Labels', 'Count']

    # Tworzenie wykresu bąbelkowego
    fig = px.scatter(
        label_counts,
        x=[1] * len(label_counts),  # Stała pozycja na osi X
        y='Label',
        size='Count',
        size_max=60,
        labels={'y': 'Etykiety', 'x': ''},
        title=title,
        hover_data=['Count']
    )

    # Dostosowanie wyglądu wykresu
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(title='Etykiety'),
        showlegend=False
    )

    return fig

def label_frequencies(df, labels_column='Labels', title='Częstość występowania etykiet zapachowych'):
    """
    Tworzy wykres bąbelkowy częstości występowania etykiet zapachowych.

    Parametry:
    -----------
    df : pd.DataFrame
        DataFrame zawierający kolumnę z etykietami.
    labels_column : str, opcjonalnie
        Nazwa kolumny zawierającej etykiety (domyślnie 'Labels').
        Etykiety mogą być listami lub stringami reprezentującymi listy.
    title : str, opcjonalnie
        Tytuł wykresu.

    Zwraca:
    --------
    fig : plotly.graph_objs._figure.Figure
        Obiekt wykresu Plotly.
    """

    # Jeśli etykiety są stringami, konwertujemy je na listy
    if df[labels_column].dtype == object and df[labels_column].apply(lambda x: isinstance(x, str)).all():
        df = df.copy()
        df[labels_column] = df[labels_column].apply(ast.literal_eval)

    # Rozpakuj listy etykiet do osobnych wierszy
    exploded = df.explode(labels_column)

    # Zlicz częstość występowania każdej etykiety
    label_counts = exploded[labels_column].value_counts().reset_index()
    label_counts.columns = ['Label', 'Count']

    return label_counts






def pco_occurrence_matrix(df, labels_column='Labels', unique_labels=None):
    """Tworzy macierz współwystępowania z wizualizacją"""
    # 1. Przygotowanie danych
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    n = len(unique_labels)
    co_matrix = np.zeros((n, n), dtype=int)
    
    # 2. Wypełnianie macierzy
    for labels in df[labels_column]:
        if isinstance(labels, str):  # Konwersja stringów do list
            labels = literal_eval(labels)
        
        valid_labels = [label for label in labels if label in label2idx]
        
        for i, j in combinations(valid_labels, 2):
            idx_i, idx_j = label2idx[i], label2idx[j]
            co_matrix[idx_i, idx_j] += 1
            co_matrix[idx_j, idx_i] += 1
    
    # 3. Tworzenie DataFrame
    matrix_df = pd.DataFrame(co_matrix, index=unique_labels, columns=unique_labels)
    
    # 4. Wizualizacja
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        matrix_df,
        cmap='viridis',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Macierz współwystępowania zapachów', pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    return matrix_df

def pco_similarity_matrix(df):
    similarity = cosine_similarity(df)
    similarity_df = pd.DataFrame(similarity, index=df.index, columns=df.columns)

        # 4. Wizualizacja
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        similarity_df,
        cmap='viridis',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title('Macierz współwystępowania zapachów', pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return similarity_df




def pca_and_cluster_visualization(similarity_df, n_components=2, n_clusters=5):
    """
    Wykonuje PCA na macierzy podobieństwa, klastrowanie KMeans i wizualizuje wyniki
    
    :param similarity_df: DataFrame macierzy podobieństwa (kwadratowa, indeksy to nazwy)
    :param n_components: liczba komponentów PCA (2 lub 3)
    :param n_clusters: liczba klastrów KMeans
    """
    # 1. PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(similarity_df)
    
    # 2. KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    
    # 3. Wizualizacja
    if n_components == 2:
        plt.figure(figsize=(10,8))
        plt.scatter(coords[:,0], coords[:,1], c=clusters, cmap='tab10', s=50)
        plt.scatter(centers[:,0], centers[:,1], c='red', s=200, marker='X', label='Centroids')
        for i, label in enumerate(similarity_df.index):
            plt.text(coords[i,0], coords[i,1], label, fontsize=8)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PCA + KMeans (k={n_clusters}) - 2D')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    elif n_components == 3:
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(coords[:,0], coords[:,1], coords[:,2], c=clusters, cmap='tab10', s=50)
        ax.scatter(centers[:,0], centers[:,1], centers[:,2], c='red', s=200, marker='X', label='Centroids')
        for i, label in enumerate(similarity_df.index):
            ax.text(coords[i,0], coords[i,1], coords[i,2], label, fontsize=8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA + KMeans (k={n_clusters}) - 3D')
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("n_components musi być 2 lub 3")
    
    return clusters[0]

def plotly_pca_kmeans(similarity_df, n_components=2, n_clusters=5):
    """
    PCA + KMeans + wizualizacja w Plotly Express (2D lub 3D)
    
    :param similarity_df: macierz podobieństwa (DataFrame)
    :param n_components: liczba komponentów PCA (2 lub 3)
    :param n_clusters: liczba klastrów KMeans
    """
    # PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(similarity_df)
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(coords)
    
    # Przygotuj DataFrame do plotly
    df_plot = pd.DataFrame(coords, columns=[f'PC{i+1}' for i in range(n_components)])
    df_plot['cluster'] = clusters.astype(str)
    df_plot['label'] = similarity_df.index.astype(str)
    
    # Wizualizacja
    if n_components == 2:
        fig = px.scatter(
            df_plot,
            x='PC1', y='PC2',
            color='cluster',
            hover_name='label',
            title=f'PCA + KMeans (k={n_clusters}) - 2D'
        )
    elif n_components == 3:
        fig = px.scatter_3d(
            df_plot,
            x='PC1', y='PC2', z='PC3',
            color='cluster',
            hover_name='label',
            title=f'PCA + KMeans (k={n_clusters}) - 3D'
        )
    else:
        raise ValueError("n_components musi być 2 lub 3")
    
    fig.show()
    return coords, clusters, df_plot

if __name__ == "__main__":
    dataset = load_and_merge_data()
    print("Final dataset shape:", dataset.shape)
    print(dataset.head())