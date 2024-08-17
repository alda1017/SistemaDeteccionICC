import wfdb
import json
import sys
import pywt
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import entropy
from scipy.signal import iirnotch, find_peaks, butter, filtfilt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if len(sys.argv) > 1:
    base_name = sys.argv[1]
    path = base_name

    record = wfdb.rdrecord(path)

    number_of_signals = record.n_sig
    sampling_frequency = record.fs
    signal_names = record.sig_name
    signal_length = record.sig_len

    def apply_filters(data, notch_freq, lowpass_freq, highpass_freq, fs, notch_Q=30, lowpass_order=1, highpass_order=5):
        b_notch, a_notch = iirnotch(notch_freq, notch_Q, fs)
        filtered_data = filtfilt(b_notch, a_notch, data)
       
        nyquist_freq = 0.5 * fs
        norm_highpass_freq = highpass_freq / nyquist_freq
        b_high, a_high = butter(highpass_order, norm_highpass_freq, btype='high', analog=False)
        filtered_data = filtfilt(b_high, a_high, filtered_data)
       
        norm_lowpass_freq = lowpass_freq / nyquist_freq
        b_low, a_low = butter(lowpass_order, norm_lowpass_freq, btype='low', analog=False)
        filtered_data = filtfilt(b_low, a_low, filtered_data)
       
        return filtered_data

    def shannon_entropy(signal):
        signal = signal + abs(np.min(signal))
        signal = signal / np.sum(signal)
        return entropy(signal)

    def wavelet_entropy(signal, wavelet='db4', level=4):
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        energy = [np.sum(np.square(c)) for c in coeffs]
        total_energy = np.sum(energy)
        entropy_wavelet = [-e/total_energy * np.log2(e/total_energy) 
                           if e != 0 else 0 for e in energy]
        return np.sum(entropy_wavelet)

    fs = record.fs

    notch_freq = 60
    lowpass_freq = 50
    highpass_freq = 0.5

    signal = record.p_signal[:, 0]

    ecg_filtered = apply_filters(signal, notch_freq, lowpass_freq, highpass_freq, fs)

    peak_distance = int(0.6 * fs)
    peak_height = np.max(ecg_filtered) * 0.5

    all_peaks, _ = find_peaks(ecg_filtered, distance=peak_distance, height=peak_height)

    sig_len = len(ecg_filtered)
    box_len = int(fs * 1)
    half_box_len = box_len // 2

    threshold_percentage = 0.1

    columnas = ['QRS_width', 'T_wave_amplitude', 'Shannon_entropy', 'Wavelet_entropy', 'Energy_scale']
    df = pd.DataFrame(columns=columnas)

    for i, sample in enumerate(all_peaks):
        start_index = max(sample - half_box_len, 0)
        end_index = min(start_index + box_len, sig_len)

        segment = np.zeros(box_len)
        actual_start_index = start_index if start_index + box_len <= sig_len else sig_len - box_len
        segment[:] = ecg_filtered[actual_start_index:end_index]

        distance = int(0.25 * fs)
        peaks, _ = find_peaks(segment, distance=distance)
        peaks = peaks[segment[peaks] > 0.8 * np.max(segment)]

        if peaks.size > 0:
            peak_ampli = segment[peaks]
            peak_amplitude = peak_ampli[0]
            threshold = peak_amplitude * threshold_percentage

            start_qrs = np.where(segment[:peaks[0]] < threshold)[0][-1] if np.any(segment[:peaks[0]] < threshold) else 0

            end_qrs = peaks[0] + np.where(segment[peaks[0]:] < threshold)[0][0] if np.any(
                segment[peaks[0]:] < threshold) else len(segment)

            qrs_width_samples = end_qrs - start_qrs

            qrs_width_secs = qrs_width_samples / fs
           
            def find_t_wave(segment, end_qrs, fs, start_offset, end_offset):
               
                start_search_t_wave = end_qrs + int(start_offset * fs)
                end_search_t_wave = end_qrs + int(end_offset * fs)
                search_window = segment[start_search_t_wave:end_search_t_wave]

                peaks_t_wave_pos, _ = find_peaks(search_window, height=None)
                if peaks_t_wave_pos.size > 0:
                    t_wave_peak_index_pos = peaks_t_wave_pos[np.argmax(search_window[peaks_t_wave_pos])] + start_search_t_wave
                    t_wave_amplitude_pos = segment[t_wave_peak_index_pos]
                else:
                    t_wave_peak_index_pos = 0
                    t_wave_amplitude_pos = 0

                inverted_window = -search_window
                peaks_t_wave_neg, _ = find_peaks(inverted_window, height=None)
                if peaks_t_wave_neg.size > 0:
                    t_wave_peak_index_neg = peaks_t_wave_neg[np.argmax(inverted_window[peaks_t_wave_neg])] + start_search_t_wave
                    t_wave_amplitude_neg = segment[t_wave_peak_index_neg]
                else:
                    t_wave_peak_index_neg = 0
                    t_wave_amplitude_neg = 0

                if t_wave_amplitude_pos != 0 and t_wave_amplitude_neg != 0:
                    if abs(t_wave_amplitude_pos) > abs(t_wave_amplitude_neg):
                        t_wave_amplitude = t_wave_amplitude_pos
                        t_wave_peak_index = t_wave_peak_index_pos
                    else:
                        t_wave_amplitude = t_wave_amplitude_neg
                        t_wave_peak_index = t_wave_peak_index_neg
                elif t_wave_amplitude_pos != 0:
                    t_wave_amplitude = t_wave_amplitude_pos
                    t_wave_peak_index = t_wave_peak_index_pos
                elif t_wave_amplitude_neg != 0:
                    t_wave_amplitude = t_wave_amplitude_neg
                    t_wave_peak_index = t_wave_peak_index_neg
                else:
                    t_wave_amplitude = 0
                    t_wave_peak_index = 0

                return t_wave_amplitude, t_wave_peak_index
           
            t_wave_amplitude, t_wave_peak_index = find_t_wave(segment, end_qrs, fs, 0.2, 0.4)
           
            if t_wave_amplitude == 0 or (t_wave_amplitude > -0.15 and t_wave_amplitude < 0.1):
                t_wave_amplitude, t_wave_peak_index = find_t_wave(segment, end_qrs, fs, 0.1, 0.2)
               
            shannon_ent = shannon_entropy(segment)

            wavelet_ent = wavelet_entropy(segment)
   
            scales = np.arange(1, 128)
            coefficients, frequencies = pywt.cwt(segment, scales, 'cmor')
           
            band_energy = np.sum(np.abs(coefficients[0:127])**2)
 
            features_df = {
                'QRS_width': qrs_width_secs,
                'T_wave_amplitude': t_wave_amplitude,
                'Shannon_entropy': shannon_ent,
                'Wavelet_entropy': wavelet_ent,
                'Energy_scale': band_energy
            }
           
            df = pd.concat([df, pd.DataFrame([features_df])], ignore_index=True)
           
    X_new = df.values

    filename = './server/Python/SVM_model.sav'

    loaded_model = pickle.load(open(filename, 'rb'))

    predictions = loaded_model.predict(X_new)

    counting_predictions = Counter(predictions)
    percentage = (counting_predictions[1] / len(predictions)) * 100

    threshold = 80

    half_index = len(ecg_filtered) // 2
    half_signal = ecg_filtered[:half_index]

    result = "Normal" if percentage < threshold else "ICC"

    output = {
        "result": result,
        "signal": half_signal.tolist(),
        "number_of_signals": number_of_signals,
        "sampling_frequency": sampling_frequency,
        "signal_names": signal_names,
        "signal_length": signal_length
    }

    print(json.dumps(output))

else:
    print("No se proporcionó el nombre del archivo.")