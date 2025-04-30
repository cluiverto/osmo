import pyrfume
import pandas as pd
import plotly.express as px
import ast

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
    label_counts.columns = ['Label', 'Count']

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


if __name__ == "__main__":
    dataset = load_and_merge_data()
    print("Final dataset shape:", dataset.shape)
    print(dataset.head())