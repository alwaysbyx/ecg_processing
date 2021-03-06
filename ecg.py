import pywt
import scipy.signal
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from signal_filter import signal_filter
from signal_delineator import dwt_ecg_delineator
from signal_delineat_byx import denpendent_delineate,indenpendent_delineate
import plotly.graph_objects as go
from scipy import stats
cmp = ['red','orange','yellow','green','cyan',
      'blue','purple','pink','magenta','brown']

def ecg_delineate(file,lead='all',method='base',sample_rate=500,plot=True,subplot=True):
    if lead=='all':
        if method=='raw':
            signals, rpeak = avg_segment(file, method='average', plot=False,clean=False,sampling_rate=sample_rate)
        else:
            signals, rpeak = avg_segment(file, method='average', plot=False,  sampling_rate=sample_rate)
        location = denpendent_delineate(signals,rpeak,sampling_rate=sample_rate, plot=plot, subplot=subplot)
        return location, signals
    else:
        _ = segment(file,lead=lead,method=method,sampling_rate=sample_rate,plot=False)
        signals, rpeak = _['epochs'], _['sr']
        location = indenpendent_delineate(signals, rpeak, sampling_rate=sample_rate, plot=plot)
        return location, signals


def delineate(file,lead='I',start=None,end=None,location=None,sampling_rate=500,plot_type='line'):
    if start==None:
        start = 0
    if end==None:
        end = len(file[lead])
    ecgsignal = np.array(file[lead][start:end])
    x = np.array(range(start, end))
    rpeaks = R_peak(file,lead=lead,sampling_rate=sampling_rate)
    rpeaks = rpeaks[rpeaks>start]
    rpeaks = rpeaks[rpeaks < end]
    if location is None:
        r = dwt_ecg_delineator(ecgsignal, rpeaks, sampling_rate=sampling_rate, analysis_sampling_rate=2000)
    else:
        r = location.copy()
        for key in r.keys():
            r[key] = np.array(r[key])
            r[key] = r[key][r[key] > start]
            r[key] = r[key][r[key] < end]
    ymin = min(ecgsignal) - 10
    ymax = max(ecgsignal) + 10
    plt.figure(figsize=(12, 5))
    plt.title(f'wave lead:{lead}')
    plt.plot(x, ecgsignal)
    if plot_type == 'line':
        i = 0
        for key in r.keys():
            plt.vlines(np.array(r[key]),ymin,ymax,color=cmp[i], linestyles="dashed", label=key)
            i += 1
    elif plot_type == 'node':
        i = 0
        for key in r.keys():
            plt.scatter(np.array(r[key]), ecgsignal[np.array(r[key])-start], color=cmp[i], linestyles="dashed", label=key, s=100)
            i += 1
    plt.legend()
    plt.show()
    return r

def filtered(file,lead='I',sampling_rate=500,plot=False):
    '''
    :param file:
    :param lead: one lead of 12 leads
    :param sampling_rate: Hz aka samplings per second
    :param plot: whether to plot the image
    :return:
        dataframe with new column 'filtered' to find R peaks
    '''
    df = file.copy()
    ecg_signal = np.array(df[lead])
    order = int(0.3 * sampling_rate)
    if order % 2 == 0:
        order += 1  # Enforce odd number
    # -> filter_signal()
    frequency = [3, 45]
    #   -> get_filter()
    #     -> _norm_freq()
    frequency = 2 * np.array(frequency) / sampling_rate  # Normalize frequency to Nyquist Frequency (Fs/2).
    #     -> get coeffs
    a = np.array([1])
    b = scipy.signal.firwin(numtaps=order, cutoff=frequency, pass_zero=False)
    # _filter_signal()
    filtered = scipy.signal.filtfilt(b, a, ecg_signal)
    df['filtered'] = pd.Series(filtered)
    if plot:
        plt.figure(figsize=(12,5))
        plt.plot(df[lead],label=lead)
        plt.plot(df['filtered'],color='green',label='filtered')
        plt.legend()
        plt.show()
    return df


# ??????R???
def R_peak(file,sampling_rate=500,lead='I',plot=False,debug=False):
    def _get_mean_top_n(data,n):
        return np.mean(data[np.argpartition(data, n)[-n:]])
    df = file.copy()
    if lead in ['V1','V2','V3']:
        df[lead] = df[lead].map(lambda x:-x)
    df = filtered(df,lead)
    df['filtered'] = df['filtered'].map(lambda x: np.abs(x))

    df['gradient'] = abs(np.gradient(df['filtered']))
    clean = baseline_drift(df,lead)['clean']
    R_peak_min = max(df['gradient']) * 0.35
    r = max(df['filtered'])*0.3
    #print(sampling_rate)
    T1 = 0.22 * sampling_rate
    T = T1
    t = 0.04 * sampling_rate
    #print(R_peak_min)
    #df['I_1'] = pd.Series(np.gradient(df['I']))
    #df['I_2'] = pd.Series(np.gradient(df['I_1']))
    df['R'] = df['gradient'].map(lambda x: R_peak_min if x > R_peak_min else 0)
    peaks = []
    peak = []
    #r_idx = []
    le = 6
    flag = True
    for idx, v in enumerate(df['R']):
        if v==R_peak_min:
            if len(peak)==0:
                peak.append(idx)
            else:
                if idx > np.mean(peak[-1])+T1:
                    if debug:
                        print('next peak',idx)
                    if len(peak) >= 1:
                        seg = pd.Series(df['filtered'][max(0,peak[0]-10):min(peak[-1]+30,len(df['filtered']))])
                        maxidx = seg.idxmax()
                        minidx = seg.idxmin()
                        #print('maxidx',maxidx)
                        if debug:
                            if idx == 2527:
                                print(f'peaks[-1]+T - maxidx = {peak[-1]} + {T} - {maxidx} = {peaks[-1]+T - maxidx}')
                                print(f'min={min(seg)}, max={max(seg)},start = {peak[0]},end={peak[-1]+30}')
                        if (len(peaks)==0 or peaks[-1]+T < maxidx) and (max(seg) > r):
                            peaks.append(maxidx)
                            if len(peaks) > 1:
                                T = int((peaks[-1] - peaks[0]) / (len(peaks) - 1) / 2.2) #?????????????????????
                    peak = []
                else:
                    if df['filtered'][idx] > r:
                        peak.append(idx)

        #if v==R_peak_min:
        #    r_idx.append(idx)
    if len(peak)>=2:
        maxidx = pd.Series(df[lead][peak[0]:peak[-1]]).idxmax()
        if peaks[-1]+T < maxidx:
            peaks.append(maxidx)
    if plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=file[lead],mode='lines',name=lead))
        if debug:
            fig.add_trace(go.Scatter(y=df['filtered'], mode='lines', name='filtered'))
            fig.add_trace(go.Scatter(y=df['R'],mode='lines', name='gradient'))
            fig.add_hline(y=r)
        fig.add_trace(go.Scatter(y=clean, mode='lines', name='baseline_drift'))
        for p in peaks:
            fig.add_vline(x=p, line_dash="dash", annotation_text="peak")
        fig.show()
    return np.array(peaks)

def baseline_drift(file,lead='I',sampling_rate=500,plot=False):
    df = file.copy()
    ecg_signal = np.array(df[lead])
    clean = signal_filter(signal=ecg_signal, sampling_rate=sampling_rate, lowcut=0.5, method="butterworth", order=5)
    clean = signal_filter(signal=clean, sampling_rate=sampling_rate, method="powerline", powerline=50)
    df['clean'] = pd.Series(clean)
    if plot:
        plt.figure(figsize=(12, 5))
        plt.plot(df[lead], label=lead)
        plt.plot(df['clean'], label='baseline_drift')
        plt.legend()
        plt.show()
    return df


def avg_segment(file,method='average',lead='all',clean=True,sampling_rate=500,plot=True):
    df = file.copy()
    # ????????????????????????
    rpeaks_lst = []
    for column in ['I','II']:
        rpeaks = R_peak(df, sampling_rate=sampling_rate, lead=column, plot=False)
        rpeaks_lst.append(len(rpeaks))
    if abs(rpeaks_lst[0] - rpeaks_lst[1]) <= 1:
        pass
    else:
        rpeaks_lst = []
        rpeaks_store = []
        for column in df.columns:
            rpeaks = R_peak(df, sampling_rate=sampling_rate, lead=column, plot=False)
            rpeaks_lst.append(len(rpeaks))
            rpeaks_store.append(rpeaks)

        idx = np.where(rpeaks_lst ==stats.mode(rpeaks_lst)[0][0])[0][0]
        #print(idx)
        rpeaks = rpeaks_store[idx]
        del rpeaks_store

    rpeaks0 = rpeaks[:-1]
    rpeaks1 = rpeaks[1:]
    idxpred = int(0.5 * np.mean(rpeaks1 - rpeaks0))
    if lead=='all':
        if plot:
            fig = go.Figure()
        r = {}
        for lead in df.columns:
            if clean == True:
                df = baseline_drift(df, lead, sampling_rate)
                ecg_signal = df['clean']
            else:
                ecg_signal = df[lead]
            epochs_start, epochs_end, epochs,_ = _ecg_segment_window(
                ecg_signal=ecg_signal, rpeaks=rpeaks, sampling_rate=sampling_rate
            )
            x_idx = np.linspace(epochs_start, epochs_end, len(epochs[0]))
            if method == 'average':
                avg = np.mean(epochs,axis=0)
            elif method == 'med':
                avg = np.median(epochs,axis=0)
            r[lead.upper()] = avg
            if plot:
                fig.add_trace(go.Scatter(x=x_idx, y=avg, mode='lines', name=lead))
        if plot:
            fig.show()
        return r, idxpred
    elif lead in df.columns:
        if clean == True:
            df = baseline_drift(df, lead, sampling_rate)
            ecg_signal = df['clean']
        else:
            ecg_signal = df[lead]
        epochs_start, epochs_end, epochs,_ = _ecg_segment_window(
            ecg_signal=ecg_signal, rpeaks=rpeaks, sampling_rate=sampling_rate
        )
        x_idx = np.linspace(epochs_start, epochs_end, len(epochs[0]))
        if method == 'average':
            avg = np.mean(epochs, axis=0)
        elif method == 'med':
            avg = np.median(epochs, axis=0)
        if plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_idx,y=avg,mode='lines',name=lead))
            fig.show()
        return avg, idxpred

def segment(file, method='base', lead='I', locate=False, r=None,sampling_rate=500, plot=True):
    df = file.copy()
    if method=='base':
        df = baseline_drift(df, lead, sampling_rate) #df['clean']
        ecgsignal = df['clean']
    elif method=='strengthened':
        df = filtered(df,lead,sampling_rate)
        ecgsignal = df['filtered']
    elif method=='raw':
        ecgsignal = df[lead]
    rpeaks = R_peak(df, sampling_rate=sampling_rate, lead=lead, plot=False)
    epochs_start, epochs_end, epochs, sr = _ecg_segment_window(
        ecg_signal=ecgsignal, rpeaks=rpeaks,sampling_rate=sampling_rate
    )
    x_idx = np.linspace(epochs_start,epochs_end,len(epochs[0]))
    if r is None:
        r = dwt_ecg_delineator(ecgsignal, rpeaks, sampling_rate=sampling_rate, analysis_sampling_rate=2000)
    if plot:
        plt.figure(figsize=(12,5))
        plt.xlabel("Time (s)")
        plt.title("Individual Heart Beats")
        for i in range(len(epochs)):
            plt.plot(x_idx, epochs[i],label="lead:%s num:%d"%(lead,i))
        if locate:
            i = 0
            for feature in r.keys():
                new,x = _delineate_transform(r[feature],feature[:1],epochs_start,epochs_end,epochs)
                plt.scatter(new,ecgsignal[x],color=cmp[i],label=feature, s=100)
                i += 1
        plt.legend(loc=2, bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
        plt.title("segmented heartbeats using %s" % method)
        plt.show()
    return {'epochs_start':epochs_start, 'epochs_end':epochs_end, 'epochs':epochs, 'location_info':r,'sr':sr}

def _delineate_transform(x,feature,start,end,epochs):
    New = np.array(x,dtype='float32')
    feature = feature.upper()
    y1 = start
    y2 = end
    w = (y2-y1)/len(epochs[0])
    xr = 0
    er = 0
    res = []
    res_pre = []
    while (xr<len(x) and er<len(epochs)):
        #print(f'f={feature},xr={xr},er={er}')
        if feature=='P':
            s = epochs[er].index[0]
            e = epochs[er].index[-1]
            e = s+(e-s)/2
        elif feature == 'T':
            s = epochs[er].index[0]
            e = epochs[er].index[-1] + 20
            s = e - (e-s)/2
        else:
            s = epochs[er].index[0]
            e = epochs[er].index[-1]
        if New[xr] >= s and New[xr] <= e:
            New[xr] = x[xr] - epochs[er].index[0]
            New[xr] = New[xr] * w + y1
            res.append(New[xr])
            res_pre.append(x[xr])
            xr += 1
            er += 1
        elif New[xr] < s:
            xr += 1
        elif New[xr] > e:
            er += 1
    return np.array(res), np.array(res_pre)


def _ecg_segment_window(ecg_signal,rpeaks, sampling_rate=500):
    # Extract heart rate
    rpeaks0 = rpeaks[:-1]
    rpeaks1 = rpeaks[1:]
    heart_rate = sampling_rate / np.mean(rpeaks1-rpeaks0) * 60

    # Modulator
    m = heart_rate / 60

    # Window
    epochs_start = -0.5 / m
    epochs_end = 0.65 / m

    idx_pre = int(0.5 * np.mean(rpeaks1-rpeaks0))
    idx_after =int(0.65 * np.mean(rpeaks1-rpeaks0))

    # Adjust for high heart rates
    if heart_rate >= 80:
        c = 0.1
        epochs_start = epochs_start - c
        epochs_end = epochs_end + c

    store_epochs = []
    store_rpeaks = []
    for p in rpeaks:
        s = p - idx_pre
        e = p + idx_after
        if s>=0 and e<len(ecg_signal):
            store_epochs.append([s,e])
            store_rpeaks.append(idx_pre)
    epochs = []
    for se in store_epochs:
        epochs.append(pd.Series(ecg_signal[se[0]:se[1]]))

    return epochs_start, epochs_end, epochs,store_rpeaks

def raise_error(ecg_signal,sampling_rate=500,plot=False):
    """

    :param ecg_signal:
    :param sampling_rate:
    :param plot:
    :return: return True if the signal is invalid else False
    """
    starts = len(ecg_signal) // sampling_rate
    for start in range(starts):
        flag =  _raise_error_window(ecg_signal,start,sampling_rate=sampling_rate,plot=plot)
        if flag==True:
            return True
    if plot:
        plt.plot(ecg_signal)
        plt.show()
    return False

def _raise_error_window(ecg_signal,start,sampling_rate=500,plot=False):
    window_size = sampling_rate #1s?????????????????????????????????
    t = np.arange(start,start+1,1.0/sampling_rate)
    window1 = ecg_signal[window_size*start:window_size*(start+1)]
    window2f = np.fft.rfft(window1)/window_size
    freqs = np.linspace(0,sampling_rate/2,window_size/2+1)
    xfp = 20*np.log10(np.clip(np.abs(window2f),1e-20,1e100))
    mean = np.mean(ecg_signal)
    var = np.var(ecg_signal)
    up = mean + 3 * var
    low = mean - 3 * var
    #print(xfp)
    #print(np.gradient(xfp))
    #print(np.var(xfp))
    #print(np.var(np.gradient(xfp)))
    if plot:
        plt.figure(figsize=(8,4))
        plt.subplot(211)
        plt.plot(t, window1)
        plt.xlabel(u"Time(S)")
        plt.title(u"ecg signal via time")
        plt.subplot(212)
        plt.plot(freqs, xfp)
        plt.xlabel(u"Freq(Hz)")
        plt.subplots_adjust(hspace=0.4)
        plt.show()
    if len(ecg_signal[ecg_signal>up]) > 0:
        return True
    if len(ecg_signal[ecg_signal<low]) > 0:
        return True
    if np.var(np.gradient(xfp)) < 1 and (max(xfp)>30 or min(xfp)<-100):
        print('var',np.var(np.gradient(xfp)))
        return True
    if max(xfp[:30]) > 40:
        return True
    return False

