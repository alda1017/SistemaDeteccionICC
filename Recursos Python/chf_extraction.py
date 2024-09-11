import wfdb
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from scipy.signal import iirnotch, lfilter, find_peaks, butter, filtfilt

# Read the file
record_name = 'C:\\Users\\alda7\\Desktop\\mew data\\segmentos2024\\recortes_listos\\icc\\segmento_chf03'
record = wfdb.rdrecord(record_name)
annotations = wfdb.rdann(record_name, 'atr')

# Information of the file
print("Number of signals:", record.n_sig)
print("Signal sampling frequency:", record.fs)
print("Signal names:", record.sig_name)
print("Signal length:", record.sig_len)

CSV_FILE_PATH = 'C:\\Users\\alda7\\Desktop\\mew data\\final_features_2.csv'

def create_dataframe(features):
    df = pd.read_csv(CSV_FILE_PATH)
    # Create a new DataFrame from features
    new_row = pd.DataFrame([features])
    # Use concat to add the new row
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)

def apply_filters(data, notch_freq, lowpass_freq, highpass_freq, fs, notch_Q=30, lowpass_order=1, highpass_order=5):
    # Apply Notch filter
    b_notch, a_notch = iirnotch(notch_freq, notch_Q, fs)
    filtered_data = filtfilt(b_notch, a_notch, data)
    
    # Apply high-pass filter with filtfilt for zero-phase
    nyquist_freq = 0.5 * fs
    norm_highpass_freq = highpass_freq / nyquist_freq
    b_high, a_high = butter(highpass_order, norm_highpass_freq, btype='high', analog=False)
    filtered_data = filtfilt(b_high, a_high, filtered_data)
    
    # Apply low-pass filter with filtfilt for zero-phase
    norm_lowpass_freq = lowpass_freq / nyquist_freq
    b_low, a_low = butter(lowpass_order, norm_lowpass_freq, btype='low', analog=False)
    filtered_data = filtfilt(b_low, a_low, filtered_data)
    
    return filtered_data


def shannon_entropy(signal):
    signal = signal + abs(np.min(signal))
    signal = signal / np.sum(signal)
    return entropy(signal)


# Función para calcular la entropia wavelet
def wavelet_entropy(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    energy = [np.sum(np.square(c)) for c in coeffs]
    total_energy = np.sum(energy)
    entropy_wavelet = [-e/total_energy * np.log2(e/total_energy) if e != 0 else 0 for e in energy]
    return np.sum(entropy_wavelet)


# SIGNAL SAMPLING RATE
fs = record.fs

# FILTER SETUP
notch_freq = 60  # Frequency to eliminate with the Notch filter
lowpass_freq = 50  
highpass_freq = 0.5 

# FILTER ANNOTATIONS TO ONLY HAVE THE N's
indices_N = [i for i, symbol in enumerate(annotations.symbol) if symbol == 'N']
samples_N = np.array(annotations.sample)[indices_N]
print(len(samples_N))

# CALCULATE RR INTERVALS
rr_intervals_samples = np.diff(samples_N)  # Differences between consecutive R peaks in samples
rr_intervals_seconds = rr_intervals_samples / fs
print("Intervalos RR en segundos:", rr_intervals_seconds)

annotations_N = wfdb.Annotation(
    record_name=record_name,
    extension='atr',
    sample=samples_N,
    symbol=['N'] * len(samples_N),
    chan=[0] * len(samples_N),
    num=[annotations.num[i] for i in indices_N],
    fs=fs
)

signal_original = record.p_signal[:, 0]
signal = -signal_original

# Apply all filters
ecg_filtered = apply_filters(signal, notch_freq, lowpass_freq, highpass_freq, fs)

# Boxcar window application
sig_len = len(ecg_filtered)
box_len = int(fs * 1)
half_box_len = box_len // 2

# threshold
threshold_percentage = 0.1  # 10%

# Process each segment
for i, sample in enumerate(samples_N):
    start_index = max(sample - half_box_len, 0)
    end_index = min(start_index + box_len, sig_len)

    # Extract the signal segment
    segment = np.zeros(box_len)
    actual_start_index = start_index if start_index + box_len <= sig_len else sig_len - box_len
    segment[:] = ecg_filtered[actual_start_index:end_index]
    
    distance = int(0.25 * fs)
    peaks, _ = find_peaks(segment, distance=distance)
    peaks = peaks[segment[peaks] > 0.8 * np.max(segment)]

    if peaks.size > 0:
        print("Extraction")
        
        if i >= len(rr_intervals_seconds):
            continue
        
        #---------------------------------------------------------------------
        # CARACTERISTICA 1 / AMPLITUD DEL PICO
        #---------------------------------------------------------------------
        peak_ampli = segment[peaks]
        peak_amplitude = peak_ampli[0]
        #---------------------------------------------------------------------
        
        #---------------------------------------------------------------------
        # CARACTERISTICA 2 / Ancho del complejo QRS
        #---------------------------------------------------------------------
        threshold = peak_amplitude * threshold_percentage
        start_qrs = np.where(segment[:peaks[0]] < threshold)[0][-1] if np.any(segment[:peaks[0]] < threshold) else 0
        end_qrs = peaks[0] + np.where(segment[peaks[0]:] < threshold)[0][0] if np.any(segment[peaks[0]:] < threshold) else len(segment)
        # Calculate the width of the QRS complex in samples
        qrs_width_samples = end_qrs - start_qrs
        
        # Convert QRS width to time seconds
        qrs_width_secs = qrs_width_samples / fs
        print(f"Ancho del QRS (muestras): {qrs_width_samples}")
        print(f"#3 Ancho del QRS (segundos): {qrs_width_secs}")
        #---------------------------------------------------------------------
        
        #---------------------------------------------------------------------
        # CARACTERISTICA X / AMPLITUD ONDA T /
        #---------------------------------------------------------------------
        # rango de busqueda para la onda T
        start_search_t_wave = end_qrs + int(0.2 * fs)
        end_search_t_wave = end_qrs + int(0.4 * fs)
        search_window = segment[start_search_t_wave:end_search_t_wave]
        
        # Invertir la señal para detectar valles como picos
        inverted_window = -search_window
        
        # Detectar picos en la señal invertida
        peaks_t_wave, _ = find_peaks(inverted_window, height=None)
        
        if peaks_t_wave.size > 0:
            t_wave_peak_index = peaks_t_wave[np.argmax(inverted_window[peaks_t_wave])] + start_search_t_wave
            t_wave_amplitude = segment[t_wave_peak_index]
        else:
            t_wave_peak_index = 0
            t_wave_amplitude = 0
            
        print(f"#4 Wave t amplitude: {t_wave_amplitude}")
        
        # Calcular la entropia de Shannon del segmento
        shannon_ent = shannon_entropy(segment)
        print(f"#5 Entropía de Shannon: {shannon_ent}")
        
        # Calcular la entropia wavelet del segmento
        wavelet_ent = wavelet_entropy(segment)
        print(f"#7 Entropía Wavelet: {wavelet_ent}")
        
        #---------------------------------------------------------------------
        # CARACTERISTICA / distribución de energía en diferentes frecuencias
        #---------------------------------------------------------------------
        
        # Calcular el escalograma
        scales = np.arange(1, 128)
        coefficients, frequencies = pywt.cwt(segment, scales, 'cmor')
        
        band_energy = np.sum(np.abs(coefficients[0:127])**2)
        
        print(f"Energy_scale: {band_energy}")
        
        # Graficar el escalograma
        plt.figure(figsize=(10, 4))
        plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto',
                    vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
        plt.title('Escalograma')
        plt.ylabel('Escalas')
        plt.xlabel('Tiempo (s)')
        plt.colorbar(label='Magnitud')
        plt.show()
        
        
        #----------------------------------------------------------------------
        # CASO DE ANALISIS DE LOS PICOS / ANOTACION V
        #----------------------------------------------------------------------
        # 0 para sano    /    1 para enfermo
        case = 1
        
        
        # ---------------------------------------------------------------------
        # Creacion del DATA FRAME
        # ---------------------------------------------------------------------
        # features_df = {
        #     'QRS_width': qrs_width_secs,
        #     'T_wave_amplitude': t_wave_amplitude,
        #     'Shannon_entropy': shannon_ent,
        #     'Wavelet_entropy': wavelet_ent,
        #     'Energy_scale': band_energy,
        #     'Class': case
        # }
        
        # # Guardar el DataFrame en un archivo CSV
        # create_dataframe(features_df)
        # ---------------------------------------------------------------------







