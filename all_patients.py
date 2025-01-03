import os
import pandas as pd

# funkcja do obliczania średnich dla inhale i exhale
def calculate_means(df):
    means = {}
    for col in ["AMP_not_corr", "SLOPE_not_corr", "AMP_corr", "SLOPE_corr", "AUC_corr"]:
        means[f"mean_{col}_inhale"] = df[df['Phase'].str.lower() == 'inhale'][col].mean()
        means[f"mean_{col}_exhale"] = df[df['Phase'].str.lower() == 'exhale'][col].mean()
    return means

# ścieżka do folderu z plikami
folder_path = "/Users/helenatryk/Desktop/ICP_data/Phases_analyse/Outputs"

# lista do przechowywania średnich dla wszystkich pacjentów
all_patients_means = []

# iteracja po wszystkich plikach w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # wczytaj plik CSV
        df = pd.read_csv(file_path)

        # oblicz średnie wartości
        means = calculate_means(df)

        # dodaj identyfikator pacjenta (zakładam, że jest w nazwie pliku np. "PAC01_output.csv")
        patient_id = filename.split("_")[0]
        means["patient_id"] = patient_id

        # dodaj średnie do listy wszystkich pacjentów
        all_patients_means.append(means)

        # dodaj nowe kolumny do DataFrame z pojedynczą wartością dla wszystkich wierszy
        for col, value in means.items():
            if col != "patient_id":
                df[col] = [value] * len(df)

        # zapisz zaktualizowany plik
        df.to_csv(file_path, index=False)

# utwórz DataFrame ze średnimi dla wszystkich pacjentów
all_patients_df = pd.DataFrame(all_patients_means)

# przekształć DataFrame
columns_inhale = [col for col in all_patients_df.columns if "_inhale" in col]
columns_exhale = [col for col in all_patients_df.columns if "_exhale" in col]

# uporządkuj kolumny
ordered_columns = ["patient_id"] + columns_inhale + columns_exhale
all_patients_df = all_patients_df[ordered_columns]

# dodaj nagłówek
inhale_header = ["inhale"] * len(columns_inhale)
exhale_header = ["exhale"] * len(columns_exhale)
multilevel_columns = pd.MultiIndex.from_tuples(
    [("patient_id", "")] + list(zip(inhale_header, columns_inhale)) + list(zip(exhale_header, columns_exhale))
)

all_patients_df.columns = multilevel_columns

# zapisz do pliku "all_patients.csv" w tym samym folderze
all_patients_df.to_csv(os.path.join(folder_path, "all_patients_2.csv"))

print("Przetwarzanie plików zakończone.")
