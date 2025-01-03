import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, detrend, iirfilter, filtfilt
from scipy.fft import fft, fftfreq
import os
import csv
from tqdm import tqdm


def convert_datetime_to_time(datetime, multi_day=True):
    '''
    Konwersja czasu z timestampów na wektor czasu w sekundach. Otrzymana od dr Kazimierskiej.
    :param datetime: wartości daty i godziny (timestampy)
    :param multi_day: flaga logiczna wskazująca na zakres wartości datetime (jeden lub więcej dni)
    :return: t_hat - tablica czasu w sekundach, fs_hat - częstotliwość próbkowania
    '''
    if not multi_day:  # analiza jesli zapis trwa jedna dobe
        t0 = (datetime[0] - np.floor(datetime[0])) * 24 * 3600  # np.floor zwraca max liczbe calkowita </= danej liczbie
        t_hat = np.squeeze((datetime - np.floor(datetime)) * 24 * 3600 - t0)
        fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
    else:  # wiecej niz jedna doba
        n_datetime = datetime - datetime[0]
        n_datetime_days = np.floor(n_datetime)  # liczba pełnych dni
        c_datetime = n_datetime - n_datetime_days  # czas w ciągu dnia
        c_datetime_seconds = c_datetime * 24 * 3600  # przeliczenie czasu dziennego na sekundy

        t_hat = []  # lista na czas w sekundach
        for idx in range(0, len(datetime)):
            c_t = n_datetime_days[idx] * 24 * 3600 + c_datetime_seconds[idx]  # liczba pełnych dni w sekundach
                                                                              # + czas dzienny w sekundach
            t_hat.append(c_t)
        t_hat = np.asarray(t_hat)
        fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)  # częstotliwość próbkowania to różnicy między
                                                      # pierwszymi dwiema wartościami czasu
    return t_hat, fs_hat


def divide_into_chunks(data_table):
    '''
    Funkcja dzieląca cały zapis sygnału na dwuminutowe fragmenty dla czytelniejszej analizy.
    :param data_table: zabiór danych zapisu sygnału
    :return: zwraca dwuminutowe fragmenty danych typu pandas.DataFrame
    '''
    start_time = data_table['Time'].iloc[0]
    chunk_length_seconds = 120
    current_time_limit = start_time + chunk_length_seconds

    chunks = []
    current_chunk = []

    for idx, row in tqdm(data_table.iterrows()):
        time_in_seconds = row["Time"]
        if time_in_seconds < current_time_limit:
            current_chunk.append(row)
        else:
            if current_chunk:
                chunks.append(pd.DataFrame(current_chunk, columns=data_table.columns))
            current_chunk = [row]
            current_time_limit += chunk_length_seconds

    if current_chunk:
        chunks.append(pd.DataFrame(current_chunk, columns=data_table.columns))
    return chunks


def filter_to_respiratory_component(time_vector, icp_signal, fs, filter_type='cheby1', order=3, freq_range=(0.1, 0.3)):
    """
    Funkcja do projektowania filtra pasmowoprzepustowego.
    :param time_vector: wektor czasu
    :param icp_signal: sygnał ICP
    :param fs: częstotliwość próbkowania
    :param filter_type: typ filtra
    :param order: rząd filtra
    :param freq_range: zakres częstotliwości składowej oddechowej
    :return: przefiltrowany sygnał
    """
    nyquist = fs / 2
    Wn = [f / nyquist for f in freq_range]

    b, a = iirfilter(
        N=order,
        Wn=Wn,
        rp=1,
        rs=20,
        btype='bandpass',
        ftype=filter_type
    )

    detrended_signal = detrend(icp_signal)
    filtered_signal = filtfilt(b, a, detrended_signal)

    return filtered_signal


def analyze_respiratory_component(time_vector, icp_signal, fs, freq_range=(0.1, 0.3)):
    """
   Funkcja analizująca składową oddechową w zadanym zakresie częstotliwości przy użyciu transformacji Fouriera.
   :param time_vector: wektor czasu
   :param icp_signal: sygnał ICP
   :param fs: częstotliwość próbkowania
   :param freq_range: zakres częstotliwości składowej oddechowej
   :return: zwraca amplitudę i częstotliwość składowej oddechowej
   """
    fft_values = fft(icp_signal)
    fft_frequencies = fftfreq(len(icp_signal), 1 / fs)  # nowa zmienna dla kolumny z icp

    freq_min, freq_max = freq_range
    filtered_indices = (fft_frequencies >= freq_min) & (fft_frequencies <= freq_max)
    filtered_fft_values = fft_values[filtered_indices]
    filtered_fft_frequencies = fft_frequencies[filtered_indices]

    if len(filtered_fft_values) == 0:
        print("No frequencies in the specified range. Skipping this chunk.")
        return None, None

    amplitudes = np.abs(filtered_fft_values)
    A = amplitudes.max() / len(icp_signal) * 2
    f = filtered_fft_frequencies[amplitudes.argmax()]

    return A, f


def process_all_files(base_folder, output_base_folder):
    for root, dirs, files in os.walk(base_folder):  # os.walk() funkcja petli po plikach
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_folder)
                output_subfolder = os.path.join(output_base_folder, relative_path)

                os.makedirs(output_subfolder, exist_ok=True)

                print(f"Processing file: {file_path}")
                data_table = pd.read_csv(file_path, delimiter=',', low_memory=False)  # wczytuje dane

                # sprawdzanie nazwy kolumny, poniewaz roznia sie miedzy soba w plikach
                if 'icp[mmHg]' in data_table.columns:
                    icp_column = 'icp[mmHg]'
                elif 'icp' in data_table.columns:
                    icp_column = 'icp'
                else:
                    print(f"Warning: File {file_path} does not contain 'icp[mmHg]' or 'icp' column.")
                    continue

                # usun nany
                data_table = data_table.dropna(subset=[icp_column])

                datetime_vector = data_table['DateTime'].values
                time_vector, fs = convert_datetime_to_time(datetime_vector)

                data_table['Time'] = time_vector


                chunks = divide_into_chunks(data_table)
                for i, chunk in enumerate(chunks[:-1]):
                    if len(chunk) < 2:
                        continue

                    time = chunk['Time'].values
                    signal = chunk[icp_column].values  # zmienna `icp_column` zawiera obie mozliwe nazwy dla kolumny
                    filtered_signal = filter_to_respiratory_component(time, signal, fs)

                    respiratory_amplitude, respiratory_frequency = analyze_respiratory_component(time, filtered_signal, fs, freq_range=(0.1, 0.3))

                    if respiratory_amplitude is None or respiratory_frequency is None:
                        print(f"Chunk {i+1} skipped due to insufficient frequency data.")
                        continue

                    A, f = analyze_respiratory_component(time, filtered_signal, fs)
                    max_peaks, _ = find_peaks(filtered_signal)
                    min_peaks, _ = find_peaks(-filtered_signal)

                    # charakterystyka i sinus
                    plt.figure(figsize=(12, 6))
                    plt.plot(time, signal, 'b', alpha=0.6, label="Sygnał ICP")
                    plt.plot(time, filtered_signal, 'r', label="Składowa oddechowa")
                    plt.plot(time[max_peaks], filtered_signal[max_peaks], '.g', label="Maksima")
                    plt.plot(time[min_peaks], filtered_signal[min_peaks], '.k', label="Minima")
                    plt.legend()
                    plt.grid()
                    plt.xlabel("Czas [s]")
                    plt.ylabel("ICP [mmHg]")
                    plot_path = os.path.join(output_subfolder, f"Chunk_{i}.png")
                    plt.savefig(plot_path, dpi=100)
                    plt.close()

                    # zapis wynikow do csv
                    csv_path = os.path.join(output_subfolder, f"Chunk_{i}.csv")
                    with open(csv_path, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["A", "f"])
                        writer.writerow([A, f])
                        writer.writerow(["Max Peaks", "Min Peaks"])
                        writer.writerows(zip(max_peaks, min_peaks))


# sciezki
base_folder = "/Users/helenatryk/Desktop/ICP_data/Data_TBI/Data_Phases"  # folder z danymi wejściowymi
output_base_folder = "/Users/helenatryk/Desktop/ICP_data/Data_TBI/Data_Phases/Data_Phases_Processed"  # folder z danymi wyjściowymi


process_all_files(base_folder, output_base_folder)  # tylko pierwszy zapis od kazdego pacjenta
