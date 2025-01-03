import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import iirfilter, filtfilt, butter, cheby1, cheby2, ellip, detrend
from tqdm import tqdm
from scipy.fft import fft, fftfreq


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
    # zakres f dzielony przez Nyguista do projektowania filtrow cyfrowych
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

    detrended_signal = detrend(icp_signal)  # usuwanie trendu z sygnału
    print("Trend removed from signal")

    filtered_signal = filtfilt(b, a, detrended_signal)  # filtracja sygnalu po usunieciu trendu
    print("ICP signal filtering completed")

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
    #  transformacja Fouriera
    fft_values = fft(icp_signal)
    fft_frequencies = fftfreq(len(signal), 1 / fs)

    #  filtracja do zakresu częstotliwości
    freq_min = freq_range[0]
    freq_max = freq_range[1]
    filtered_indices = (fft_frequencies >= freq_min) & (fft_frequencies <= freq_max)
    filtered_fft_values = fft_values[filtered_indices]
    filtered_fft_frequencies = fft_frequencies[filtered_indices]

    amplitudes = np.abs(filtered_fft_values)
    A = amplitudes.max() / len(icp_signal) * 2  # max amplituda
    f = filtered_fft_frequencies[amplitudes.argmax()]  # częstotliwość jej odpowiadająca

    return A, f


file_path = "/Users/helenatryk/Desktop/ICP_data/Data_TBI/PAC10/PAC10_r01.csv"
data_table = pd.read_csv(file_path, delimiter=',', low_memory=False)

print("Data has been imported")

datetime_vector = data_table['DateTime'].values  # wyciągnięcie kolumny DateTime (timestampy)
time_vector, fs = convert_datetime_to_time(datetime_vector)  # zmiana timestampów na wektor czasu w sekundach
data_table['Time'] = time_vector  # dodanie nowej kolumny z wektorem czasu

print(f"Changed timestamps to time vector: {time_vector[:5]}")
print(f"Sampling frequency: {fs}")

print(f"The number of 'NaN' in a column 'icp[mmHg]': {data_table['icp[mmHg]'].isna().sum()}")

data_table = data_table.dropna(subset=['icp[mmHg]'])  # usunięcie NaNów z dataframe'a

print(f"Removed 'NaN' values")

time_vector = data_table['Time'].values  # pomocniczo wyciągnięcie kolumny z wektorem czasu na potrzeby sprawdzania długości zapisu
icp_signal = data_table['icp[mmHg]'].values  # pomocniczo wyciągnięcie kolumny z sygnałem ICP

start_time = time_vector[0]  # znalezienie czasu początkowego (pierwsza próbka wektora czasu)
end_time = time_vector[-1]  # znalezienie czasu końcowego (ostatnia próbka wektora czasu)

if end_time - start_time > 1 * 60 * 60:  # sprawdzenie, czy długość zapisu jest większa niż 1 h
    end_row = int(1 * 60 * 60 * fs)  # jeżeli tak - docięcie do 1 h (indeksowanie zakończone na 1 h * częstotliwość próbkowania)
else:
    end_row = len(time_vector)  # jeżeli nie - zostawienie obecnej długości zapisu (indeksowanie zakończone na obecnej długości zapisu)

data_table = data_table.head(end_row)  # docięcie zapisu do ustalonej powyżej długości

print(f"Extracted the first hour of recording")

chunks = divide_into_chunks(data_table)

print("Division into fragments has been completed")
print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:-1]):  # pomijamy ostatni chunk, bo może być za krótki do filtracji
    print(f"Chunk {i+1}: number of rows = {len(chunk)}")
    if len(chunk) < 2:
        print("Chunk skipped: not enough data")
        continue

    time = chunk['Time'].values
    signal = chunk['icp[mmHg]'].values

    filtered_signal = filter_to_respiratory_component(time, signal, fs, filter_type='cheby1', order=3, freq_range=(0.1, 0.3))
    respiratory_amplitude, respiratory_frequency = analyze_respiratory_component(time, signal, fs, freq_range=(0.1, 0.3))
    print(f"Analyzed fundamental respiratory component")
    print(f"Amplitude: {respiratory_amplitude}, frequency: {respiratory_frequency} Hz")

    respiratory_cycle = 1 / respiratory_frequency
    min_distance = 0.75 * respiratory_cycle  # minimalna odległość pomiędzy pikami (3/4 długości cyklu oddechowego)
    min_height = 0.5 * respiratory_amplitude  # minimalna wysokość piku (połowa amplitudy składowej oddechowej)

    max_peaks, _ = find_peaks(filtered_signal, distance=min_distance, height=min_height)  # maksima oddechowe
    min_peaks, _ = find_peaks(-filtered_signal, distance=min_distance, height=min_height)  # minima oddechowe

    plt.figure(figsize=(12, 6))
    plt.plot(time, signal, 'b', label='Pełny sygnał ICP',  alpha=0.6)
    plt.plot(time, filtered_signal, 'b', label='Składowa oddechowa (0.1-0.3 Hz)', linewidth=2)
    plt.plot(time[max_peaks], filtered_signal[max_peaks], '.r', label='Maksima oddechowe')
    plt.plot(time[min_peaks], filtered_signal[min_peaks], '.k', label='Minima oddechowe')
    plt.xlabel('Czas [s]')
    plt.ylabel('Sygnał ICP [mm Hg]')
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Chunk {i+1} has been processed.")

print("Processing completed.")
