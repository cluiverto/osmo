import pyrfume
import pandas as pd

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


if __name__ == "__main__":
    dataset = load_and_merge_data()
    print("Final dataset shape:", dataset.shape)
    print(dataset.head())