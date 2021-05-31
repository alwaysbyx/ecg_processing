import numpy as np
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import tqdm
import pywt
import scipy.signal
import scipy.ndimage
import tqdm
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

def indenpendent_delineate(signals, rpeaks, sampling_rate=500, plot=True):
    locations = {}
    if plot:
        fig = go.Figure()
    for i in range(len(signals)):
        try:
            signal = np.array(signals[i])
            rpeak = rpeaks[i]
            if signal[rpeak] < 0:
                gra1 = np.gradient(signal)
                if gra1[rpeak] < 0:
                    mingra1 = np.argmin(gra1)
                    for idx in range(mingra1, mingra1 // 2, -1):
                        if gra1[idx] < 0 and gra1[idx - 1] > 0:
                            rpeak = idx
                            break
            Q = _delineate_Q(signal, rpeak)
            Ppeak = _delineate_Ppeak(signal, rpeak)
            Poff = _delineate_Poff(signal, rpeak)
            Pon = _delineate_Pon(signal, rpeak)
            QRSon = _delineate_QRSon(signal, rpeak, Poff)
            if QRSon is np.nan:
                QRSon = Q
            QRSoff = _delineate_QRSoff(signal, rpeak, QRSon, Pon)
            Tpeak = _delineate_Tpeak(signal, QRSoff)
            Toff = _delineate_Toff(signal, rpeak, QRSoff, Tpeak)
            location = {"Poff": Poff, "Pon": Pon, "QRSon": QRSon, "QRSoff": QRSoff, "Toff": Toff, "Rpeak": rpeak}
            locations[i] = location
            if plot == True:
                x = np.linspace(0, len(signal), len(signal))
                show_node = [Poff, Pon, QRSon, Toff, Toff, QRSoff]
                fig.add_trace(go.Scatter(x=x, y=signal,
                                         mode='lines',
                                         name=i))
                fig.add_vline(x=show_node[0], line_dash="dash", annotation_text="Poff")
                fig.add_vline(x=show_node[1], line_dash="dash", annotation_text="Pon")
                fig.add_vline(x=show_node[2], line_dash="dash", annotation_text="QRSon")
                fig.add_vline(x=show_node[-1], line_dash="dash", annotation_text="QRSoff")
                fig.add_vline(x=show_node[3], line_dash="dash", annotation_text="Toff")
        except Exception as e:
            print(i,e)
    if plot:
        fig.show()
    return locations


def denpendent_delineate(signals, rpeak, sampling_rate=500, plot=True, subplot=True):
    """
    :param signals: signal including I,II,V1,V2,V5
    :param rpeak: rpeak of average signal
    :param sampling_rate: Hz
    :return: Pon,Poff,QRSon,QRSoff,Toff
    """
    r1 = signals['I']
    r2 = signals['II']
    v1 = signals['V1']
    v2 = signals['V2']
    v3 = signals['V3']
    v4 = signals['V4']
    v5 = signals['V5']
    if r2[rpeak] < 0:
        gra1 = np.gradient(r2)
        if gra1[rpeak] < 0:
            mingra1 = np.argmin(gra1)
            for idx in range(mingra1,mingra1//2,-1):
                if gra1[idx] < 0 and gra1[idx-1] > 0:
                    rpeak = idx
                    break
    Q = _delineate_Q(r1, rpeak)
    Ppeak = _delineate_Ppeak(r2, rpeak)
    Poff = _delineate_Poff(r1, rpeak)
    Pon = _delineate_Pon(r2, rpeak)
    QRSon = _delineate_QRSon(v3, rpeak, Poff)
    if QRSon is np.nan:
        QRSon = Q
    QRSoff = _delineate_QRSoff(v5, rpeak, QRSon, Pon)
    Tpeak_v3 = _delineate_Tpeak(v3, QRSoff)
    Toff_v3 = _delineate_Toff(v3, rpeak, QRSoff, Tpeak_v3)
    Tpeak_v2 = _delineate_Tpeak(v2, QRSoff)
    Toff_v2 = _delineate_Toff(v2, rpeak, QRSoff, Tpeak_v2)
    Toff = int(np.mean([Toff_v2,Toff_v3]))
    location = {"Poff":Poff,"Pon":Pon,"QRSon":QRSon,"QRSoff":QRSoff,"Toff":Toff,"Rpeak":rpeak}
    if plot==True:
        Y = [r1, r2, v1, v2, v3, v5, v4]
        x = np.linspace(0, len(r1), len(r1))
        name = ['I', 'II', 'V1', 'V2', 'V3', 'V5','V4']
        node_name = ['Poff', 'Pon', 'QRSon', 'Toff', 'Toff', 'QRSoff']
        show_node = [Poff, Pon, QRSon, Toff, Toff, QRSoff]
        if subplot==True:
            _plot_delineate(x, Y, name,show_node,node_name,subplot=True)
        else:
            _plot_delineate(x, signals, name, show_node, node_name, subplot=False)
    return location

def _plot_delineate(x,Y,name,show_node,node_name,subplot=True):
    if subplot==True:
        fig = make_subplots(rows=len(name), cols=1, shared_xaxes=True, vertical_spacing=0)
        for i in range(len(name)):
            fig.append_trace(go.Scatter(x=x, y=Y[i],
                                        mode='lines',
                                        name=name[i]), row=i + 1, col=1)
            if i <= 5:
                fig.append_trace(go.Scatter(x=[show_node[i]], y=[Y[i][show_node[i]]],
                                            mode='markers',
                                            name=node_name[i]), row=i + 1, col=1)
            if i == 0:
                fig.add_vline(x=show_node[0], line_dash="dash", annotation_text="Poff")
                fig.add_vline(x=show_node[1], line_dash="dash", annotation_text="Pon")
                fig.add_vline(x=show_node[2], line_dash="dash", annotation_text="QRSon")
                fig.add_vline(x=show_node[-1], line_dash="dash", annotation_text="QRSoff")
                fig.add_vline(x=show_node[3], line_dash="dash", annotation_text="Toff")
            else:
                fig.add_vline(x=show_node[0], line_dash="dash")
                fig.add_vline(x=show_node[1], line_dash="dash")
                fig.add_vline(x=show_node[2], line_dash="dash")
                fig.add_vline(x=show_node[-1], line_dash="dash")
                fig.add_vline(x=show_node[3], line_dash="dash")
        fig.update_layout(height=1200, width=600, title_text="Delineate")
        fig.show()
    else:
        fig = go.Figure()
        for i in Y.keys():
            fig.add_trace(go.Scatter(x=x, y=Y[i],
                                     mode='lines',
                                     name=i))
        fig.add_vline(x=show_node[0], line_dash="dash", annotation_text="Poff")
        fig.add_vline(x=show_node[1], line_dash="dash", annotation_text="Pon")
        fig.add_vline(x=show_node[2], line_dash="dash", annotation_text="QRSon")
        fig.add_vline(x=show_node[-1], line_dash="dash", annotation_text="QRSoff")
        fig.add_vline(x=show_node[3], line_dash="dash", annotation_text="Toff")
        fig.show()

def _delineate_Poff(seg,rpeak):
    ppeak = _delineate_Ppeak(seg,rpeak)
    Q = _delineate_Q(seg,rpeak)
    end = Q-(Q-ppeak)//3
    gra1 = np.gradient(seg)
    gra2 = np.gradient(gra1)
    gra2peak = np.argmax(gra2[ppeak:end])+ppeak
    idx = gra2peak
    return idx

def _delineate_Tpeak(seg,QRSoff):
    seg1 = signal.detrend(seg[QRSoff:])
    gra1 = np.gradient(seg1)
    gra2 = np.gradient(gra1)
    for start in range(0,len(gra1)): #sampling_rate
        if gra2[start] < 0.1 and gra2[start] > -0.1:
            break
    #print('start',start)
    seg = signal.detrend(seg[QRSoff+start:])
    if np.argmax(gra1[start:]) > np.argmin(gra1[start:]):
        #print('min')
        idx = np.argmin(seg) + QRSoff +start
    else:
        #print('max')
        idx = np.argmax(seg) + QRSoff + start
    return idx

def _delineate_Ppeak(seg,rpeak):
    Q = _delineate_Q(seg,rpeak)
    seg = signal.detrend(seg[:Q])
    #print('Q',Q)
    idx = np.argmax(seg[Q//3:Q]) + Q//3
    return idx

def _delineate_Pon(seg,rpeak):
    ppeak = _delineate_Ppeak(seg,rpeak)
    #print('ppeak',ppeak)
    gra1 = np.gradient(seg)
    gra2 = np.gradient(gra1)
    delta = gra2-gra1
    minidx = np.argmin(delta[ppeak//2:ppeak]) + ppeak//2
    idx = minidx - 15 #delay
    return idx

def _delineate_Toff(seg,rpeak,QRSoff,Tpeak):
    gra1 = np.gradient(seg)
    gra2 = np.gradient(gra1)
    yuzhi= 0.2
    count = 0
    for idx in range(Tpeak,len(seg)):
        if gra1[idx] < yuzhi and gra1[idx] > -yuzhi and idx > Tpeak+20: #sampling_rate=500
            count += 1
            if count == 2:
                break
    return idx

def _delineate_QRSoff(seg,rpeak,QRSon,Pon):
    t = (rpeak - QRSon) * 2
    gra1 = np.gradient(seg)
    gra2 = np.gradient(gra1)
    gra3 = np.gradient(gra2)
    gra2bottom = np.argmin(gra2)
    gra2peak = np.argmax(gra2[gra2bottom:])+gra2bottom
    idx = np.argmin(gra2[gra2peak+1:gra2peak+t])+gra2peak
    return idx+3 #delay


def _delineate_QRSon(seg,rpeak,Poff):
    gra1 = np.gradient(seg)
    gra2 = np.gradient(gra1)
    #gra3 = np.gradient(gra2)
    end = np.argmin(gra1)
    gra2peak = np.argmax(gra2[Poff:end]) + Poff
    idx = gra2peak
    for idx in range(gra2peak,Poff,-1):
        if gra2[idx] < 0.2:
            break
    if idx == gra2peak:
        return np.nan
    return idx

def _delineate_Q(seg,rpeak):
    gra1 = np.gradient(seg)
    gra2 = np.gradient(gra1)
    idx = rpeak
    for idx in range(rpeak-2,rpeak//2,-1):
        if gra1[idx] > 0 and gra1[idx-1] < 0:
            break
    return idx