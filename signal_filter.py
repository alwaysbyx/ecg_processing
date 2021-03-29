import pandas as pd
import numpy as np
import scipy.signal
"""apply from
https://github.com/neuropsychology/NeuroKit/blob/883bbb3461d1e40dc5f16a1f6eb2a7babb9f0758/neurokit2/signal/signal_filter.py#L264"""

def signal_filter(
    signal,
    sampling_rate=1000,
    lowcut=None,
    highcut=None,
    method="butterworth",
    order=2,
    window_size="default",
    powerline=50,
):
    method = method.lower()
    if method in ["powerline"]:
        filtered = _signal_filter_powerline(signal, sampling_rate, powerline)
    elif method in ["butter", "butterworth"]:
        filtered = _signal_filter_butterworth(signal, sampling_rate, lowcut, highcut, order)
    return filtered

def _signal_filter_butterworth(signal, sampling_rate=500, lowcut=None, highcut=None, order=5):
    """Filter a signal using IIR Butterworth SOS method."""
    freqs, filter_type = _signal_filter_sanitize(lowcut=lowcut, highcut=highcut, sampling_rate=sampling_rate)

    sos = scipy.signal.butter(order, freqs, btype=filter_type, output="sos", fs=sampling_rate)
    filtered = scipy.signal.sosfiltfilt(sos, signal)
    return filtered

def _signal_filter_powerline(signal, sampling_rate, powerline=50):
    """Filter out 50 Hz powerline noise by smoothing the signal with a moving average kernel with the width of one
    period of 50Hz."""

    if sampling_rate >= 100:
        b = np.ones(int(sampling_rate / powerline))
    else:
        b = np.ones(2)
    a = [len(b)]
    y = scipy.signal.filtfilt(b, a, signal, method="pad")
    return y

def _signal_filter_sanitize(lowcut=None, highcut=None, sampling_rate=1000, normalize=False):
    # Replace 0 by none
    if lowcut is not None and lowcut == 0:
        lowcut = None
    if highcut is not None and highcut == 0:
        highcut = None

    # Format
    if lowcut is not None and highcut is not None:
        if lowcut > highcut:
            filter_type = "bandstop"
        else:
            filter_type = "bandpass"
        freqs = [lowcut, highcut]
    elif lowcut is not None:
        freqs = [lowcut]
        filter_type = "highpass"
    elif highcut is not None:
        freqs = [highcut]
        filter_type = "lowpass"

    # Normalize frequency to Nyquist Frequency (Fs/2).
    # However, no need to normalize if `fs` argument is provided to the scipy filter
    if normalize is True:
        freqs = np.array(freqs) / (sampling_rate / 2)

    return freqs, filter_type

def signal_zerocrossings(signal, direction="both"):
    """Locate the indices where the signal crosses zero.
    Note that when the signal crosses zero between two points, the first index is returned.
    Parameters
    ----------
    signal : Union[list, np.array, pd.Series]
        The signal (i.e., a time series) in the form of a vector of values.
    direction : str
        Direction in which the signal crosses zero, can be "positive", "negative" or "both" (default).
    Returns
    -------
    array
        Vector containing the indices of zero crossings.
    Examples
    --------
    """
    df = np.diff(np.sign(signal))
    if direction in ["positive", "up"]:
        zerocrossings = np.where(df > 0)[0]
    elif direction in ["negative", "down"]:
        zerocrossings = np.where(df < 0)[0]
    else:
        zerocrossings = np.nonzero(np.abs(df) > 0)[0]

    return zerocrossings