import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tqdm
import pywt
import scipy.signal
import scipy.ndimage
from signal_filter import signal_zerocrossings

# =============================================================================
# WAVELET METHOD (DWT)
# =============================================================================

def _resample_interpolation(signal, desired_length=None, sampling_rate=None, desired_sampling_rate=None):
    if desired_length is None:
        desired_length = int(np.round(len(signal) * desired_sampling_rate / sampling_rate))
    resampled_signal = scipy.ndimage.zoom(signal, desired_length / len(signal))
    return resampled_signal


def _dwt_resample_points(peaks, sampling_rate, desired_sampling_rate):
    """Resample given points to a different sampling rate."""
    if isinstance(peaks, np.ndarray):  # peaks are passed in from previous processing steps
        # Prevent overflow by converting to np.int64 (peaks might be passed in containing np.int32).
        peaks = peaks.astype(dtype=np.int64)
    elif isinstance(peaks, list):  # peaks returned from internal functions
        # Cannot be converted to int since list might contain np.nan. Automatically cast to np.float64 if list contains np.nan.
        peaks = np.array(peaks)
    peaks_resample = peaks * desired_sampling_rate / sampling_rate
    peaks_resample = [np.nan if np.isnan(x) else int(x) for x in peaks_resample.tolist()]
    return peaks_resample


def dwt_ecg_delineator(ecg, rpeaks, sampling_rate=500, analysis_sampling_rate=2000):
    """Delinate ecg signal using discrete wavelet transforms.
    Parameters
    ----------
    ecg : Union[list, np.array, pd.Series]
        The cleaned ECG channel as returned by `ecg_clean()`.
    rpeaks : Union[list, np.array, pd.Series]
        The samples at which R-peaks occur. Accessible with the key "ECG_R_Peaks" in the info dictionary
        returned by `ecg_findpeaks()`.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
    analysis_sampling_rate : int
        The sampling frequency for analysis (in Hz, i.e., samples/second).
    Returns
    --------
    dict
        Dictionary of the points.
    """
    ecg = _resample_interpolation(ecg, sampling_rate=sampling_rate, desired_sampling_rate=analysis_sampling_rate)
    dwtmatr = _dwt_compute_multiscales(ecg, 9)

    # # only for debugging
    # for idx in [0, 1, 2, 3]:
    #     plt.plot(dwtmatr[idx + 3], label=f'W[{idx}]')
    # plt.plot(ecg, '--')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    rpeaks_resampled = _dwt_resample_points(rpeaks, sampling_rate, analysis_sampling_rate)

    tpeaks, ppeaks = _dwt_delineate_tp_peaks(ecg, rpeaks_resampled, dwtmatr, sampling_rate=analysis_sampling_rate)
    qrs_onsets, qrs_offsets = _dwt_delineate_qrs_bounds(
        rpeaks_resampled, dwtmatr, ppeaks, tpeaks, sampling_rate=analysis_sampling_rate
    )
    ponsets, poffsets = _dwt_delineate_tp_onsets_offsets(ppeaks, dwtmatr, sampling_rate=analysis_sampling_rate)
    tonsets, toffsets = _dwt_delineate_tp_onsets_offsets(
        tpeaks, dwtmatr, sampling_rate=analysis_sampling_rate, onset_weight=0.6, duration=0.6
    )

    waves = dict(
        P_Onsets=_dwt_resample_points(ponsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        P_Peaks=_dwt_resample_points(ppeaks, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        P_Offsets=_dwt_resample_points(poffsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        Q=_dwt_resample_points(qrs_onsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        R_peaks=rpeaks,
        S=_dwt_resample_points(qrs_offsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        T_Onsets=_dwt_resample_points(tonsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        T_Peaks=_dwt_resample_points(tpeaks, analysis_sampling_rate, desired_sampling_rate=sampling_rate),
        T_Offsets=_dwt_resample_points(toffsets, analysis_sampling_rate, desired_sampling_rate=sampling_rate)

    )
    # Remove NaN in Peaks, Onsets, and Offsets
    waves_noNA = waves.copy()
    for feature in waves_noNA.keys():
        waves_noNA[feature] = [int(x) for x in waves_noNA[feature] if ~np.isnan(x)]

    return waves_noNA


def _dwt_compensate_degree(sampling_rate):
    return int(np.log2(sampling_rate / 250))


def _dwt_delineate_tp_peaks(
        ecg,
        rpeaks,
        dwtmatr,
        sampling_rate=500,
        qrs_width=0.13,
        p2r_duration=0.2,
        rt_duration=0.25,
        degree_tpeak=3,
        degree_ppeak=2,
        epsilon_T_weight=0.25,
        epsilon_P_weight=0.02,
):
    srch_bndry = int(0.5 * qrs_width * sampling_rate)
    degree_add = _dwt_compensate_degree(sampling_rate)

    tpeaks = []
    for rpeak_ in rpeaks:
        if np.isnan(rpeak_):
            tpeaks.append(np.nan)
            continue
        # search for T peaks from R peaks
        srch_idx_start = rpeak_ + srch_bndry
        srch_idx_end = rpeak_ + 2 * int(rt_duration * sampling_rate)
        dwt_local = dwtmatr[degree_tpeak + degree_add, srch_idx_start:srch_idx_end]
        height = epsilon_T_weight * np.sqrt(np.mean(np.square(dwt_local)))

        if len(dwt_local) == 0:
            tpeaks.append(np.nan)
            continue

        ecg_local = ecg[srch_idx_start:srch_idx_end]
        peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
        peaks = list(filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks))  # pylint: disable=W0640
        if dwt_local[0] > 0:  # just append
            peaks = [0] + peaks

        # detect morphology
        candidate_peaks = []
        candidate_peaks_scores = []
        for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
            correct_sign = dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0  # pylint: disable=R1716
            if correct_sign:
                idx_zero = signal_zerocrossings(dwt_local[idx_peak:idx_peak_nxt + 1])[0] + idx_peak
                # This is the score assigned to each peak. The peak with the highest score will be
                # selected.
                score = ecg_local[idx_zero] - (float(idx_zero) / sampling_rate - (rt_duration - 0.5 * qrs_width))
                candidate_peaks.append(idx_zero)
                candidate_peaks_scores.append(score)

        if not candidate_peaks:
            tpeaks.append(np.nan)
            continue

        tpeaks.append(candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start)

    ppeaks = []
    for rpeak in rpeaks:
        if np.isnan(rpeak):
            ppeaks.append(np.nan)
            continue

        # search for P peaks from Rpeaks
        srch_idx_start = rpeak - 2 * int(p2r_duration * sampling_rate)
        srch_idx_end = rpeak - srch_bndry
        dwt_local = dwtmatr[degree_ppeak + degree_add, srch_idx_start:srch_idx_end]
        height = epsilon_P_weight * np.sqrt(np.mean(np.square(dwt_local)))

        if len(dwt_local) == 0:
            ppeaks.append(np.nan)
            continue

        ecg_local = ecg[srch_idx_start:srch_idx_end]
        peaks, __ = scipy.signal.find_peaks(np.abs(dwt_local), height=height)
        peaks = list(filter(lambda p: np.abs(dwt_local[p]) > 0.025 * max(dwt_local), peaks))
        if dwt_local[0] > 0:  # just append
            peaks = [0] + peaks

        # detect morphology
        candidate_peaks = []
        candidate_peaks_scores = []
        for idx_peak, idx_peak_nxt in zip(peaks[:-1], peaks[1:]):
            correct_sign = dwt_local[idx_peak] > 0 and dwt_local[idx_peak_nxt] < 0  # pylint: disable=R1716
            if correct_sign:
                idx_zero = signal_zerocrossings(dwt_local[idx_peak:idx_peak_nxt + 1])[0] + idx_peak
                # This is the score assigned to each peak. The peak with the highest score will be
                # selected.
                score = ecg_local[idx_zero] - abs(
                    float(idx_zero) / sampling_rate - p2r_duration
                )  # Minus p2r because of the srch_idx_start
                candidate_peaks.append(idx_zero)
                candidate_peaks_scores.append(score)

        if not candidate_peaks:
            ppeaks.append(np.nan)
            continue

        ppeaks.append(candidate_peaks[np.argmax(candidate_peaks_scores)] + srch_idx_start)

    return tpeaks, ppeaks


def _dwt_delineate_tp_onsets_offsets(
        peaks,
        dwtmatr,
        sampling_rate=500,
        duration=0.3,
        duration_offset=0.3,
        onset_weight=0.4,
        offset_weight=0.4,
        degree_onset=2,
        degree_offset=2,
):
    degree = _dwt_compensate_degree(sampling_rate)
    onsets = []
    offsets = []
    for i in range(len(peaks)):  # pylint: disable=C0200
        # look for onsets
        srch_idx_start = peaks[i] - int(duration * sampling_rate)
        srch_idx_end = peaks[i]
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            onsets.append(np.nan)
            continue
        dwt_local = dwtmatr[degree_onset + degree, srch_idx_start:srch_idx_end]
        onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
        if len(onset_slope_peaks) == 0:
            onsets.append(np.nan)
            continue
        epsilon_onset = onset_weight * dwt_local[onset_slope_peaks[-1]]
        if not (dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
            onsets.append(np.nan)
            continue
        candidate_onsets = np.where(dwt_local[: onset_slope_peaks[-1]] < epsilon_onset)[0]
        onsets.append(candidate_onsets[-1] + srch_idx_start)

        # # only for debugging
        # events_plot([candidate_onsets, onset_slope_peaks], dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.show()

    for i in range(len(peaks)):  # pylint: disable=C0200
        # look for offset
        srch_idx_start = peaks[i]
        srch_idx_end = peaks[i] + int(duration_offset * sampling_rate)
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            offsets.append(np.nan)
            continue
        dwt_local = dwtmatr[degree_offset + degree, srch_idx_start:srch_idx_end]
        offset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
        if len(offset_slope_peaks) == 0:
            offsets.append(np.nan)
            continue
        epsilon_offset = -offset_weight * dwt_local[offset_slope_peaks[0]]
        if not (-dwt_local[offset_slope_peaks[0]:] < epsilon_offset).any():
            offsets.append(np.nan)
            continue
        candidate_offsets = np.where(-dwt_local[offset_slope_peaks[0]:] < epsilon_offset)[0] + offset_slope_peaks[0]
        offsets.append(candidate_offsets[0] + srch_idx_start)

        # # only for debugging
        # events_plot([candidate_offsets, offset_slope_peaks], dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.show()

    return onsets, offsets


def _dwt_delineate_qrs_bounds(rpeaks, dwtmatr, ppeaks, tpeaks, sampling_rate=500):
    degree = int(np.log2(sampling_rate / 250))
    onsets = []
    for i in range(len(rpeaks)):  # pylint: disable=C0200
        # look for onsets
        srch_idx_start = ppeaks[i]
        srch_idx_end = rpeaks[i]
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            onsets.append(np.nan)
            continue
        dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
        onset_slope_peaks, __ = scipy.signal.find_peaks(-dwt_local)
        if len(onset_slope_peaks) == 0:
            onsets.append(np.nan)
            continue
        epsilon_onset = 0.5 * -dwt_local[onset_slope_peaks[-1]]
        if not (-dwt_local[: onset_slope_peaks[-1]] < epsilon_onset).any():
            onsets.append(np.nan)
            continue
        candidate_onsets = np.where(-dwt_local[: onset_slope_peaks[-1]] < epsilon_onset)[0]
        onsets.append(candidate_onsets[-1] + srch_idx_start)

        # # only for debugging
        # events_plot(candidate_onsets, -dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.legend()
        # plt.show()

    offsets = []
    for i in range(len(rpeaks)):  # pylint: disable=C0200
        # look for offsets
        srch_idx_start = rpeaks[i]
        srch_idx_end = tpeaks[i]
        if srch_idx_start is np.nan or srch_idx_end is np.nan:
            offsets.append(np.nan)
            continue
        dwt_local = dwtmatr[2 + degree, srch_idx_start:srch_idx_end]
        onset_slope_peaks, __ = scipy.signal.find_peaks(dwt_local)
        if len(onset_slope_peaks) == 0:
            offsets.append(np.nan)
            continue
        epsilon_offset = 0.5 * dwt_local[onset_slope_peaks[0]]
        if not (dwt_local[onset_slope_peaks[0]:] < epsilon_offset).any():
            offsets.append(np.nan)
            continue
        candidate_offsets = np.where(dwt_local[onset_slope_peaks[0]:] < epsilon_offset)[0] + onset_slope_peaks[0]
        offsets.append(candidate_offsets[0] + srch_idx_start)

        # # only for debugging
        # events_plot(candidate_offsets, dwt_local)
        # plt.plot(ecg[srch_idx_start: srch_idx_end], '--', label='ecg')
        # plt.legend()
        # plt.show()

    return onsets, offsets


def _dwt_compute_multiscales(ecg: np.ndarray, max_degree):
    """Return multiscales wavelet transforms."""

    def _apply_H_filter(signal_i, power=0):
        zeros = np.zeros(2 ** power - 1)
        timedelay = 2 ** power
        banks = np.r_[
            1.0 / 8, zeros, 3.0 / 8, zeros, 3.0 / 8, zeros, 1.0 / 8,
        ]
        signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
        signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 2 steps
        return signal_f

    def _apply_G_filter(signal_i, power=0):
        zeros = np.zeros(2 ** power - 1)
        timedelay = 2 ** power
        banks = np.r_[2, zeros, -2]
        signal_f = scipy.signal.convolve(signal_i, banks, mode="full")
        signal_f[:-timedelay] = signal_f[timedelay:]  # timeshift: 1 step
        return signal_f

    dwtmatr = []
    intermediate_ret = np.array(ecg)
    for deg in range(max_degree):
        S_deg = _apply_G_filter(intermediate_ret, power=deg)
        T_deg = _apply_H_filter(intermediate_ret, power=deg)
        dwtmatr.append(S_deg)
        intermediate_ret = np.array(T_deg)
    dwtmatr = [arr[: len(ecg)] for arr in dwtmatr]  # rescale transforms to the same length
    return np.array(dwtmatr)