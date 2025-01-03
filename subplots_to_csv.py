import os
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv

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


def divide_into_half_hour(data_table):
    '''
    Funkcja dzieląca cały zapis sygnału na półgodzinne fragmenty dla czytelniejszej analizy.
    :param data_table: zabiór danych zapisu sygnału
    :return: zwraca półgodzinne fragmenty danych typu pandas.DataFrame
    '''
    start_time = data_table.iloc[0, 0]  # początkowy czas w pierwszej kolumnie DateTime
    half_hour_seconds = 1800  # 30 min * 60 s = 1800 s

    chunks = []  # lista na fragmenty
    current_chunk = []  # lista na bieżące fragmenty
    current_time_limit = start_time + half_hour_seconds  # limit czasowy -> czas początkowy + 1800 s

    for idx, row in data_table.iterrows():
        time_in_seconds = row["DateTime"]  # kolumna przechowująca znaczniki czasowe w sekundach
        if time_in_seconds < current_time_limit:  # jeżeli czas nie przekracza limitu
            current_chunk.append(row)  # bieżący fragment jest dodawany do tablicy
        else:  # jeśli przekracza
            chunks.append(pd.DataFrame(current_chunk, columns=data_table.columns))  # konwersja na DataFrame
            current_chunk = [row.values]  # nowy fragment
            current_time_limit += half_hour_seconds  # zwiększanie limitu o 30 min

    if current_chunk:  # jeśłi po skończeniu pętli pozostają jeszcze dane to
        chunks.append(pd.DataFrame(current_chunk))  # zostają dodane jako ostatni chunk

    return chunks


def filter_fourier_chunks(data_table, sampling_rate, freq_min=0.1, freq_max=0.3):
    '''
    Funkcja wykorzystuje szybką transformatę fouriera (FFT) w celu wyodrębnienia częstotliwości składowej oddechowej
    w zakresie 0.1 Hz - 0.3 Hz. Funkcja filtruje fragmenty sygnału zostawiając tylko mieszczące się w podanym
    zakresie częstotliwości.
    :param data_table: zbiór danych, obiekt typu pandas.DataFrame
    :param sampling_rate: częstotliwość próbkowania sygnału na sekundę
    :param freq_min: 0.1 Hz -> dolna granica częstotliwości
    :param freq_max: 0.3 Hz -> górna granica częstotliwości
    :return: zwraca cztery listy: valid chunks -> fragmenty po filtracji, max_amplitudes -> max wartość amplitudy
    dla każdego przefiltrowanego fragmentu, mean_chunk_time -> średni czas każdego fragmentu, max_frequencies -> max
    częstotliwości dla każdej z max amplitud
    '''
    # tworzenie list dla kazdej zmiennej
    valid_chunks = []
    max_amplitudes = []
    mean_chunk_time = []
    max_frequencies = []

    chunks = divide_into_half_hour(data_table)  # pdoział na fragmenty

    for chunk in chunks:
        if len(chunk) < 2:  # jeżeli fragment ma mniej niż dwie dane
            print("Chunk skipped: not enough data")  # jest pomijany
            continue  # dalej

        signal = chunk.iloc[:, 1].values
        if np.any(np.isnan(signal)):  # isnan sprawdza fragment na obecnosc NaN
            print("Chunk skipped: NaN in chunk")  # jesli sa NaN, fragment jest pomijany
            continue

        time = chunk.iloc[:, 0].values
        chunk_time = np.mean(time)  # średni czas trwania fragmentu
        signal = chunk.iloc[:, 1].values
        fft_values = fft(signal)  # transformata fouriera
        fft_frequencies = fftfreq(len(signal), 1 / sampling_rate)  # fftfreq generuje częstotliwości

        filtered_indices = (fft_frequencies >= freq_min) & (fft_frequencies <= freq_max)  # tylko <0.1, 0.3> Hz
        if not np.any(filtered_indices):  # jeśli brak tych częstotliwości
            print("Chunk skipped: no frequencies in range")  # fragment pomijany
            continue

        filtered_fft_values = fft_values[filtered_indices]
        filtered_fft_frequencies = fft_frequencies[filtered_indices]

        amplitudes = np.abs(filtered_fft_values)

        if not np.any(np.isnan(signal)):  # dodawanie wartości do list jeśli nie ma NaN
            valid_chunks.append(chunk)
            max_amplitudes.append(amplitudes.max())
            mean_chunk_time.append(chunk_time)
            max_frequency = filtered_fft_frequencies[amplitudes.argmax()]
            max_frequencies.append(max_frequency)

    return valid_chunks, max_amplitudes, mean_chunk_time, max_frequencies


def process_file(file_path, output_folder):
    '''
    Funkcja obsługująca każdy z odczytanych plików w celu ich ewentualnej korekty do dalszej obróbki.
    :param file_path: ścieżka do pliku do przetworzenia
    :param output_folder: ścieżka folderu na wyniki (wykresy)
    '''
    try:
        data_table = pd.read_csv(file_path, delimiter=',', low_memory=False)  # wczytanie danych

        try:  # pierwsza kolumna datetime konwertowana na typ numeryczny
            data_table.iloc[:, 0] = pd.to_numeric(data_table.iloc[:, 0], errors='coerce')
        except Exception as e:  # plik pominięty z powodu problemów z konwersją kolumny
            print(f"File {file_path} skipped: problem with time column conversion. Error: {e}")
            return

        # usuwanie wierszy z błędnymi wartościami w kolumnie datetime
        data_table = data_table.dropna(subset=[data_table.columns[0]])
        if data_table.empty:  # plik pominiety z powodu braku prawidłowych danych
            print(f"File {file_path} skipped: no valid data in the time column.")
            return

        datetime_column = data_table.iloc[:, 0].values
        t_hat, fs_hat = convert_datetime_to_time(datetime_column, multi_day=True)
        data_table.iloc[:, 0] = t_hat

        sampling_rate = fs_hat
        filtered_chunks, max_amplitudes, mean_chunk_time, max_frequencies = filter_fourier_chunks(data_table, sampling_rate)

        # rysowanie wykresów i zapisywanie do plików
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        result = plot_subplots(t_hat, data_table.iloc[:, 1].values, mean_chunk_time,
                               max_amplitudes, max_frequencies, output_folder, file_name)
        if result is None:
            print(f"File {file_path} skipped: data errors")
            return
        max_amplitude, max_frequency = result

        # zapis danych do pliku CSV
        output_csv = os.path.join(output_folder, f"{file_name}_results.csv")
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)

            # nagłówki i dane do CSV
            csvwriter.writerow(["Max Amplitude", "Max Frequency"])
            csvwriter.writerow([max_amplitude, max_frequency])
            csvwriter.writerow([])
            csvwriter.writerow(["Subplot 1: Signal in Time"])
            csvwriter.writerow(["Time [s]", "Amplitude"])
            for time, amplitude in zip(t_hat, data_table.iloc[:, 1].values):
                csvwriter.writerow([time, amplitude])
            csvwriter.writerow([])
            csvwriter.writerow(["Subplot 2: Max Amplitude"])
            csvwriter.writerow(["Time [s]", "Max Amplitude"])
            for time, amplitude in zip(mean_chunk_time, max_amplitudes):
                csvwriter.writerow([time, amplitude])
            csvwriter.writerow([])
            csvwriter.writerow(["Subplot 3: Max Frequency"])
            csvwriter.writerow(["Time [s]", "Max Frequency"])
            for time, frequency in zip(mean_chunk_time, max_frequencies):
                csvwriter.writerow([time, frequency])

        print(f"Results saved in: {output_csv}")
    except Exception as e:
        print(f"File {file_path} skipped: error: {e}")


def plot_subplots(t_hat, signal, mean_chunk_time, max_amplitudes, max_frequencies, output_folder, file_name):
    """
    Funkcja rysująca wykresy i zapisująca je do pliku.
    :param t_hat: wektor czasu sygnału wejściowego
    :param signal: sygnał ICP
    :param mean_chunk_time: średni czas fragmentu
    :param max_amplitudes: maksymalne amplitudy we fragmentach
    :param max_frequencies: maksymalne częstotliwości we fragmentach
    :param output_folder: ścieżka do folderu na zapisywane pliki
    :param file_name: nazwa nowego zapisanego pliku
    :return: maksymalna wartość amplitudy i częstotliwości z list max_amplitudes i max_frequencies
    """
    plt.figure(figsize=(15, 10))

    if not max_amplitudes or not max_frequencies:
        print("Error: no data to draw graphs")
        return None, None

    # sygnał w czasie
    plt.subplot(3, 1, 1)
    plt.plot(t_hat, signal)
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")
    plt.title("Sygnał w czasie")
    plt.xlim(left=0)

    # maksymalna amplituda od mean_chunk_time
    plt.subplot(3, 1, 2)
    plt.plot(mean_chunk_time, max_amplitudes, marker='o')
    plt.xlabel("Czas [s]")
    plt.ylabel("Maksymalna amplituda [-]")
    plt.title("Maksymalna amplituda w zakresie 0.1-0.3 Hz")
    plt.xlim(left=0)
    max_amplitude = max(max_amplitudes)

    # maksymalna częstotliwość od mean_chunk_time
    plt.subplot(3, 1, 3)
    plt.plot(mean_chunk_time, max_frequencies, marker='o')
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.title("Maksymalna częstotliwość w zakresie 0.1-0.3 Hz")
    plt.xlim(left=0)
    max_frequency = max(max_frequencies)

    plt.tight_layout()

    # zapis wykresu jako plik obrazu
    output_image_path = os.path.join(output_folder, f"{file_name}_plots.png")
    plt.savefig(output_image_path, format='png')
    plt.close()  # zamknij wykres, aby nie obciążać pamięci

    print(f"Graphs saved in: {output_image_path}")
    return max_amplitude, max_frequency


# główna funkcja iteracji
def process_all_files(base_folder, output_base_folder, log_file="processed_files.log"):
    # wczytaj listę już przetworzonych plików
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            processed_files = set(log.read().splitlines())
    else:
        processed_files = set()

    # iteracja po plikach
    for root, _, files in os.walk(base_folder):  # os.walk fcja zaproponowana przez dr Kazimierska
        for file in files:
            if file.endswith(".csv"):  # Przetwarzaj tylko pliki CSV
                file_path = os.path.join(root, file)

                # sprawdź, czy plik już został przetworzony
                if file_path in processed_files:
                    print(f"Processed file skipped: {file_path}")
                    continue

                # tworzenie nowego folderu na wyniki
                relative_path = os.path.relpath(root, base_folder)
                output_folder = os.path.join(output_base_folder, relative_path)
                os.makedirs(output_folder, exist_ok=True)

                # przetwarzanie pliku
                try:
                    process_file(file_path, output_folder)

                    # dodanie pliku do logu po pomyślnym przetworzeniu
                    with open(log_file, "a") as log:
                        log.write(file_path + "\n")
                    print(f"File processing completed: {file_path}")

                except Exception as e:
                    print(f"Error during processing file {file_path}: {e}")

base_folder = "/Users/helenatryk/Desktop/ICP_data/Data_TBI"  # folder z danymi wejściowymi
output_base_folder = "/Users/helenatryk/Desktop/ICP_data/Data_TBI/Processed_Data"  # folder z danymi wyjściowymi

process_all_files(base_folder, output_base_folder)  # przetwarzanie wszystkich plików w pętli

