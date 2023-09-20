#!/usr/bin/env python3
"""
Data fit and processing for Trackable

Takes ball tracking data from the side and generates
Bounce, Line, and Speed values for app display
"""

__author__ = "Eric Habib"
__copyright__ = "Copyright 2023, Tarkett Sports Canada Inc"
__credits__ = ["Eric Habib"]
__version__ = "0.6.0"
__maintainer__ = "Eric Habib"
__email__ = "eric.habib@fieldturf.com"
__status__ = "Prototype"

import argparse
import os
from datetime import datetime
#from cv2 import DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

import numpy as np
np.seterr(all='warn')
from copy import deepcopy

import warnings
#from scipy.signal import savgol_filter
from scipy.stats import linregress as linreg
from scipy.signal import find_peaks, find_peaks_cwt
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt             # for plotting graphs
import pandas as pd
from itertools import groupby
from scipy.optimize import fsolve
from scipy.fft import fft

from threading import Thread
from datetime import datetime
import copy

import csv
from types import SimpleNamespace

from data_processing import remove_outliers
from mongo import Mongo
import json

from sys import path
import warnings

#from frame_processing.DeepSort.object_detector_and_tracker import Object_Tracker
#path.insert(1, "C:\\Users\\habibe\\Documents\\Projects\\Ball Tracking\\main_script")
#from fileutils import getFilesInFolder


# Globals
originaldir = os.getcwd()

class DataProcessor:
    """
    Class for taking raw ball detection data, processing it, detecting bounces
    and crests, fitting it to parabola.
    """

    def __init__(self, path=None, framerate=60, pathfile=None, **kwargs):
        """
        Class contructor, initializes variables and sets default smoothing

        :param data: time and position data
        :type data: list, list of lists, or pandas.DataFrame, optional
        :param framerate: Frame rate of original video, to correlate frame\
        number, and time
        :type framerate: integer, optional
        """
        if path is None:
            if pathfile is None:
                print("[Warning] DataProcessor pathfile is None")
                self.path= pd.DataFrame()
                return
                #raise RuntimeError("No data provided for processing, either provide DataFrame or data file")
            self.pathfile = pathfile
            if os.path.splitext(self.pathfile)[1] in ['.csv', '.txt', '.dat']:
                self.path = pd.read_csv(self.pathfile)
            elif os.path.splitext(self.pathfile)[1] in ['.xls', '.xlsx']:
                self.path = pd.read_excel(self.pathfile)
            else:
                raise ValueError("File type must be CSV or XLSX.")
        else:
            self.path = path  # pandas.DataFrame, raw and calculated data
        
        self.all_paths = deepcopy(self.path)
        self.active_object = None

        self.dropped = []
        self.slicer = []

        self.pathfile = pathfile
        self.framerate = framerate

        self.video_params = {}
        self.pipeline = []        # For future version modularity, not implemented

        self.features = SimpleNamespace()
        ## Step 1 - Outlier removal
        self.rm_outliers = False
        self.outliers_removed = False
        
        ## Step 2 - Smoothing and baseline adjustment
        self.smoothing = (2, 15, 5)
        self.autosmooth = True
        self.snr_signal_level = 0.6 # value between 0 and 1, lower is smoother
        
        self.force_flip=True
        self.end_baseline=True
        self.autoslice=True
        self.sliced=False

        ## Step 3 - bounce detection
        self.bouncewindow = self.framerate/3
        self.bouncedist = self.framerate/5
        self.bounce_criteria = {'position': True,
                                'velocity': True,
                                'acceleration': False}
        self.snr_thresh = 8
        self.snr_filter_thresh = 0.05
        self.bounceban = []
        self.bounceforce = []
        self.features.bounces = []

        ## Step 4 - Crest detection
        self.crestwindow = self.framerate/3
        self.crestdist = self.framerate/8
        self.crestmin = 5
        self.crestban = []
        self.crestforce = []
        self.features.crests = []

        ## Step 5 Curve Fitting
        self.features.pos_fit = None

        self.indices = None

        self.features.fit_bounces = None
        self.features.fit_crests = None

        self.features.speeds = []
        self.features.fit_speeds = []
        
        # Pixel to meter conversion
        if 'scale' in kwargs and not np.isnan(kwargs['scale']):
            self.scale = self.features.px_conv = kwargs['scale']
        else:
            self.features.px_conv = None

        self.path_warning = False

    @property
    def framerate(self):
        return self._framerate
    
    @framerate.setter
    def framerate(self, framerate):
        if framerate < 1:
            framerate = 1
        else:
            self._framerate = framerate
        if not self.path is None:
            self.path['Time'] = self.path.index / self.framerate

    @property
    def bouncemax(self):
        return self._bouncemax

    @bouncemax.setter
    def bouncemax(self, bouncemax):
        self._bouncemax = bouncemax
        self.bouncemin = self.bouncemax*0.05

    @property
    def autoslice(self):
        return self._autoslice

    @autoslice.setter
    def autoslice(self, autoslice):
        self._autoslice = autoslice
        if not autoslice and self.sliced:
            self.path = self.original_data.copy(deep=True)
            self.sliced=False

    def write_to_mongo(self):
        """
        Spawn thread to write to MongoDB database
        """
        self.mongoThread = Thread(target=self.insert_mongo_thread)
        self.mongoThread.run()

    def insert_mongo_thread(self):
        """
        Insert results from file to Trackable collection of Mongo Database
        """
        self.mongo = Mongo()
        writeable = self.build_result_obj( self.pathfile)
        self.mongo.updateRecord(writeable)

    def view_data(self, verbose=False):
        def onpick(event):
            ind = event.ind
            if not type(ind) == type(1):
                ind = ind[0]
            if verbose >= 1:
                print("Pick index:", ind)
            
            line = event.artist
            xdata, ydata = line.get_data()
            x = xdata[ind]
            y = ydata[ind]
            if verbose >= 1:
                print('on pick line:', x, y)
            if event.mouseevent.button == 1:
                print("Line chosen x:", x, ",", y, "() click, add")
                self.slicer += [int(ind)]
            elif event.mouseevent.button == 3:
                rm_ind = self.slicer.sort(key=lambda x:abs(x-ind))[0]
                if verbose >= 1:
                    print("Remove point", rm_ind)
                self.slicer.remove(rm_ind)
                    
            else:
                return
            event.artist.figure.clear()
            event.artist.axes.plot(self.path['Time'], self.path['Pos X'], picker=True)
            event.artist.axes.plot(self.path['Time'], self.path['Pos Y'], picker=True)
            if len(self.slicer) > 0:
                event.artist.axes.scatter([self.path['Time'].iloc[i] for i in self.slicer],
                            [self.path['Pos Y'].iloc[i] for i in self.slicer])

        fig, ax = plt.subplots(2)
        ax[0].plot(self.path['Time'], self.path['Pos X'], picker=True)
        ax[0].plot(self.path['Time'], self.path['Pos Y'], picker=True)
        ax[1].plot(self.path['Time'], stats.zscore(self.path['Pos X']))
        ax[1].plot(self.path['Time'], stats.zscore(self.path['Pos Y']))
        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()
        
    def export_split(self, verbose=False):
        if verbose >= 1:
            print("Export split data")
        if len(self.slicer) == 0:
            print("No split chosen, quitting")
            return
        
        self.slicer = [0] + self.slicer + [len(self.path)-1]
        
        for i in range(len(self.slicer)-1):
            self.path.loc[self.slicer[i]:self.slicer[i+1], :].to_csv("sliced_"+str(i)+".csv")
        if verbose >= 1:
            print("Done export")

    @staticmethod
    def signaltonoise(a, axis=0, ddof=0, verbose=False):
        if type(a) == pd.Series:
            x = np.arange(0, len(a), 1)
            y = np.asanyarray(a)
        else:
            x = a.iloc[:,0]
            y = a.iloc[:,1]
        
        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.plot(x, y)

        #sum residuals
        res = 0
        for st in np.arange(0,  len(a), 20):
            tx = x[st:st+20]
            ty = y[st:st+20]
            p = np.polyfit(tx, ty, 3, full=True)
            if verbose >= 1:
                ax.plot(tx, (p[0][0]*tx**3+p[0][1]*tx**2+p[0][2]*tx**1+p[0][3] ))
                #print(p)
            if len(p[1] > 0) and not np.isnan(p[1][0]):
                res += p[1][0]

        res = np.sqrt(res)
        if verbose >= 1:
            print(res)
        
        return res
    
    @staticmethod
    def noise_frequency(y,x, verbose=0):
        signal = np.fft.fft(y)
        freq = np.fft.fftfreq(len(y))

        freq_peaks = {
            'real':find_peaks(signal.real, prominence=300),
            'imag':find_peaks(signal.imag, prominence=40)
        }

        if verbose >= 1:
            fig, ax=plt.subplots()
            ax.plot(freq, signal.real, freq, signal.imag)
            ax.scatter([freq[i] for i in freq_peaks['real'][0]],
                       [signal[i] for i in freq_peaks['real'][0]])
            fig.canvas.manager.set_window_title("Fourier Analysis")
            print("FFT peaks: ", freq_peaks)

    @staticmethod
    def hypothenuse(arr):
        try:
            sumsq = sum([i**2 for i in arr])
        except:
            return None
        #print(arr, sumsq)
        return np.sqrt(sumsq)

    def evaluate_window_function(self, data=None, minval=1, maxval=31, signal_level=0.3, verbose=False):
        """
        Measure the noise of residuals after curve fitting as a method to determine optimal smoothing
        """
        if data is None:
            data = self.path.copy(deep=True)

        winvals = np.arange(minval, maxval, 2)
        residuals = []
        for i in winvals:
            if i > len(data):
                break
            sd = self.smooth(data, window_len=i)
            a = self.signaltonoise(pd.DataFrame.from_dict({'x':np.arange(0, len(sd)), 'Pos Y':sd}), verbose=False)
            residuals.append(a)
        
        # Find local minimum /critical point
        (peak, _) = find_peaks([-i for i in residuals], distance=2)
        
        # If there are no critical points, use half maximum value
        if len(peak) > 0:
            idx = peak[0]
            res = winvals[idx]
        else:
            array = np.asarray(residuals)
            idx = (np.abs(array-array.max()*signal_level)).argmin()
            res = winvals[idx]
    
        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.scatter(np.arange(1, 31, 2), residuals)
            ax.scatter(winvals[idx], residuals[idx])
            fig.canvas.set_window_title("Evaluate Window Function")
            #plt.show()
        
        return res

    @staticmethod
    def smooth(data, window_len=11, window='bartlett', verbose=False):
        """
        Smooths input values, with different algorithm types.

        :param x: Dependent variable to smooth
        :type x: list of floats
        :param window_len: Window length to apply smoothing, must have an\
        uneven value.
        :type window_len: integer, optional
        :param window: Smoothing algorithm type, from 'flat', 'hanning',\
        'hamming', 'bartlett', 'blackman'
        :type window: string, optional
        """


        if (window_len <= 1):
            return data

        if data.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if data.size < window_len:
            raise ValueError("Input vector must be bigger than window size.")

        if window_len < 3:
            return data

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming',\
             'bartlett', 'blackman'")
        
        x = data.copy(deep=True)

        s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
        if window == 'flat':
            # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.'+window+'(window_len)')

        y = np.convolve(w/w.sum(), s, mode='valid')
        y = y[int(window_len/2):-int(window_len/2)]

        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.plot(x)
            ax.plot(y)

        return y

    def remove_outliers(self, z_value=2.5, window=30, verbose=False):
        """
        Remove sharp noise by comparing stdev of previous
        """
        if self.outliers_removed:
            if verbose >= 1:
                print("Outliers already removed, skipping...")
            return

        outliers = ()
        for i in range(0, len(self.path['Pos Y'])-window, int(window/2)):
            x_win = self.path['Pos X'].iloc[i:i+window].values.tolist()
            y_win = self.path['Pos Y'].iloc[i:i+window].values.tolist()

            linearX = linreg(np.arange(len(x_win)), x_win)
            linearY = linreg(np.arange(len(y_win)), y_win)

            tx = [x_win[j] - linearX.slope*j - linearX.intercept for j in range(len(x_win))]
            ty = [y_win[j] - linearY.slope*j - linearY.intercept for j in range(len(y_win))]

            zx = np.abs(stats.zscore(tx))
            zy = np.abs(stats.zscore(ty))

            temp_out = list(filter(lambda i: i[1] > z_value, enumerate(zx + zy)))
            temp_out = [(val[0] + i, val[1]) for val in temp_out]
            outliers = tuple(list(outliers) + temp_out)

        original_data = self.path.copy(deep=True)

        self.dropped = [self.path.index[i[0]] for i in outliers]
        self.path = self.path.drop(self.dropped, axis=0)

        if verbose >= 1:
            print("All outliers: ", outliers)
            print("Dropped indices: ", self.dropped)
            f, ax = plt.subplots(1, 2)

            ax[0].plot(original_data['Pos X'], original_data['Pos Y'])
            ax[1].plot(self.path['Pos X'], self.path['Pos Y'])
            ax[1].scatter(original_data['Pos X'].loc[self.dropped[0]],
                          original_data['Pos Y'].loc[self.dropped[0]])
        
        self.outliers_removed = True

        return

    def smooth_data(self, verbose=0, autosmooth=None):
        """
        Generate the smoothed data for X, Y, as selected in the class parameters
        """
        if autosmooth is not None:
            self.autosmooth = autosmooth

        (psmooth, vsmooth, asmooth) = self.smoothing
        ####### Smoothing ########
        
        ### X Data
        psmoothx = psmoothy = psmooth
        if self.autosmooth:
            ### Autoset smoothing window
            snrx = self.evaluate_window_function(self.path['Pos X adj'], signal_level = self.snr_signal_level)
            psmoothx = snrx
        self.path['Pos X smooth'] = self.smooth(self.path['Pos X adj'], window_len=psmoothx)

        ### Y Data
        # Assume video is upright (opencv higher Y is lower in the image)
        if self.autosmooth:
            snry = self.evaluate_window_function(self.path['Pos Y adj'], signal_level = self.snr_signal_level, verbose=verbose-1)
            if verbose >= 1:
                print("Y signal to noise window set to: ", snry)
            psmoothy=snry
            self.smoothing = (snry, vsmooth, asmooth)
            #self.path['Pos X adj']
            if verbose >= 1:
                print('Y position noise level:', snry)
        self.path['Pos Y smooth'] = self.smooth(self.path['Pos Y adj'], window_len=psmoothy)

        ### Show results
        if verbose >= 1:
            if verbose == 'all':
                f, ax = plt.subplots(1,5, figsize=(14.5,5))
            else:
                f, ax = plt.subplots(2,1, figsize=(14.5,5), sharex=True)

            ax[0].plot(self.path['Time'], self.path['Pos X adj'])
            ax[0].plot(self.path['Time'],self.path['Pos X smooth'])
            ax[0].set_ylabel('X Position (px)')
            ax2 = ax[0].twinx()
            ax2.plot(self.path['Time'], -(self.path['Pos Y']-self.path['Pos Y'].min()))
            ax2.plot(self.path['Time'],self.path['Pos Y adj'])

            ax2.plot(self.path['Time'],self.path['Pos Y smooth'])
            ax2.set_ylabel('Y position (px)')
            ax[0].legend(['X', 'X smooth'])
            ax[0].set_title("Position")

            f.canvas.manager.set_window_title("Data Processing")

        # Velocity
        self.path['Vel X'] = [None] + \
            list(np.diff(self.path['Pos X adj'])*self.framerate)
        snrvx = self.evaluate_window_function(self.path['Vel X'], signal_level = self.snr_signal_level)
        self.path['Vel X smooth'] = self.smooth(
            self.path['Vel X'], window_len=snrvx)

        self.path['Vel Y'] = [None] + \
            list(np.diff(self.path['Pos Y adj'])*self.framerate)
        snrvy = self.evaluate_window_function(self.path['Vel Y'], signal_level = self.snr_signal_level)
        self.path['Vel Y smooth'] = self.smooth(
            self.path['Vel Y'], window_len=snrvy)

        if verbose >= 1:
            if verbose == 'all':
                ax[2].plot(self.path['Time'],self.path['Vel X'])
                ax[2].plot(self.path['Time'],self.path['Vel X smooth'])
                ax[2].set_title("Velocity X")
                ax[2].legend(['X', 'X smooth'])
            ax[1].plot(self.path['Time'],self.path['Vel Y'])
            ax[1].plot(self.path['Time'],self.path['Vel Y smooth'])
            ax[1].set_title("Velocity Y")
            ax[1].legend(['Y', 'Y smooth'])

        # Acceleration
        self.path['Acc X'] = [None] + \
            list(np.diff(self.path['Vel X'])*self.framerate)
        snrax = self.evaluate_window_function(self.path['Acc X'], signal_level = self.snr_signal_level)
        self.path['Acc X smooth'] = self.smooth(
            self.path['Acc X'], window_len=snrax)

        self.path['Acc Y'] = [None] + \
            list(np.diff(self.path['Vel Y'])*self.framerate)
        snray = self.evaluate_window_function(self.path['Acc Y'], signal_level = self.snr_signal_level)
        self.path['Acc Y smooth'] = self.smooth(
            self.path['Acc Y'], window_len=snray)

        if verbose == 'all':
            ax[3].plot(self.path['Time'],self.path['Acc X'])
            ax[3].plot(self.path['Time'],self.path['Acc X smooth'])
            ax[3].set_title("Acceleration X")
            ax[4].plot(self.path['Time'],self.path['Acc Y'])
            ax[4].plot(self.path['Time'],self.path['Acc Y smooth'])
            ax[4].set_title("Acceleration Y")

    def format_data(self, time=None, position=None):
        """
        Ingest and standardize data from csv to account for different possible column names
        Mostly required due to format changes over time.
        """
        if 'Frame' not in self.path.columns:
            if 'frame_num' in self.path.columns:
                self.path['Frame'] = self.path['frame_num']
            else:
                raise ValueError("Frame column not found")

        self.time = []
        
        # Allow for directly setting position
        if not position is None:
            try:
                len(position[0])
                self.path['Pos Y'] = position
            except:
                self.path['Pos X'] = position[0]
                self.path['Pos Y'] = position[1]

        # Harmonize X value column names
        if not 'Pos X' in self.path.columns:
            if 'x' in self.path.columns:
                self.path['Pos X'] = self.path['x']
            elif 'center-x' in self.path.columns:
                self.path['Pos X'] = self.path['center-x']
            elif 'left' in self.path.columns:
                self.path['Pos X'] = self.path['left']+(self.path['width']/2)
            else:
                raise ValueError("X Column format not recognized")

        # Harmonize Y value column names
        if not 'Pos Y' in self.path.columns:
            if 'y' in self.path.columns:
                self.path['Pos Y'] = self.path['y']
            elif 'center-y' in self.path.columns:
                self.path['Pos Y'] = self.path['center-y']
            elif 'top' in self.path.columns:
                self.path['Pos Y'] = self.path['top']+(self.path['height']/2)
            else:
                raise ValueError("Y Column format not recognized")

        if self.rm_outliers:
            self.remove_outliers()

        # Position
        if not time is None and not 'Time' in self.path.columns:
            self.path['Time'] = time
        else:
            self.path['Time'] = self.path.index / self.framerate
        
        self.bouncemax = (max(self.path['Pos Y'])-min(self.path['Pos Y']))/2

    @staticmethod
    def end_baseline_adjust(data, type, verbose=False):
        """
        Fix the drift in the end region of the baseline.
        Mostly needed due to camera perspective and non-orthogonality.
        """

        yonly = False
        try:
            data['Time']
        except:
            #print(data)
            yonly = True
            data = pd.DataFrame({'Time':np.arange(len(data)), 'position':data})

        ind = int(len(data)/4)
        baseline_sq = np.polyfit(data['Time'].iloc[ind:], data['position'].iloc[ind:], deg=2, full=True)
        
        if verbose >= 1:
            print(type, " baseline equation:", baseline_sq)
        
        baseline_sq = baseline_sq[0]
        if type.upper()=='Y' and (np.abs(baseline_sq[0]) > 0.1 or np.abs(baseline_sq[1]) > 1):
            warnings.warn("End Baseline is not flat enough, data may be corrupted")
        baseline = (baseline_sq[0]*data['Time']**2+baseline_sq[1]*data['Time']+baseline_sq[2])
        adjusted = data['position']-baseline
        if verbose >= 1:
            fig, ax = plt.subplots()
            
            ax.plot(data['Time'], data['position'])
            ax.plot(data['Time'], baseline)
            ax.plot(data['Time'], adjusted)
            fig.canvas.manager.set_window_title('End Baseline Adjust')
            fig.legend(['Position', 'Baseline', 'Adjusted Position'])

        return adjusted
    
    @staticmethod
    def rectify(data, order='decreasing', force_flip=False):
        """
        Rectifies (flips) input data where necessary.
        """
        if order.lower() == 'decreasing' or order.lower() == 'dec':
            rising = False
        elif order.lower() == 'increasing' or order.lower() == 'inc':
            rising = True
        else:
            raise ValueError("order value must be either '[inc]reasing' or '[dec]reasing'")
        
        slope, intercept,_,_,_ = linreg(np.arange(len(data)), data)
        if (slope > 0 and not rising) or (slope < 0 and rising) or force_flip:
            return data * -1
        return data
        
    @staticmethod
    def zero_offset(data):
        # Zero offset
        return data - np.amin(data)

    @staticmethod
    def rectify_and_baseline(data, type, force_flip=False, baseline=True, verbose=False):
        """
        Rectifies input data, zeroes offset, and subtracts baseline.
        """
        if type.upper() == 'Y':
            order='decreasing'
        elif type.upper() == 'X':
            order='increasing'
        else:
            raise ValueError("Type must be either 'X' or 'Y'")

        if baseline:
            tdata = DataProcessor.end_baseline_adjust(data, type, verbose=verbose).copy(deep=True)
        tdataf = DataProcessor.rectify(tdata, order=order, force_flip=force_flip)
        tdatafb = DataProcessor.zero_offset(tdataf)

        if verbose >= 1:
            fig, ax = plt.subplots()
            ax1 = ax.twinx()
            ax.plot(np.arange(len(data)), data)
            ax1.plot(np.arange(len(tdata)), tdata)
            ax.plot(np.arange(len(tdatafb)), tdatafb)
            
            fig.legend(['Data', 'Data BL (Ax2)', 'Data BL adj'])
            fig.canvas.manager.set_window_title("Rectify and Baseline")

        #print (tdata)
        return tdatafb

    @staticmethod
    def get_baseline_index(data=None, fraction=50, verbose=False):
        """
        Combs data for linear regions, where the ball is no longer
        bouncing.
        """

        def monoExp(x, m, t, b):
            if abs(t*max(x)) > 100:
                return 0
            try:
                #print("monoExp")
                val =  m * np.exp(-t * x) + b
                return val
            except FloatingPointError as e:
                return 0
        def getExpAtVal(val, parms):
            with warnings.catch_warnings() as w:
                temp = -np.log((val-parms[2])/parms[0])/parms[1]
                if w is not None:
                    print(f"[Warning] getExpAtVal Runtime Warning: {val}, {parms}")
            return temp

        ind = int(len(data)/4)
        try:
            data['Time']
        except:
            data = pd.DataFrame({'Time': np.arange(len(data)), 'Pos Y adj':data})

        baseline = linreg(data['Time'].iloc[ind:], data['Pos Y adj'].iloc[ind:])
        diff = (data['Pos Y adj'] - (data['Time']*baseline[0]+baseline[1])).abs()
        
        p0 = (1, 1, 1) # start with values near those we expect
        try:
            params, cv = curve_fit(monoExp, data['Time'], diff, p0)
            valatp5pc = getExpAtVal(data['Pos Y adj'].max()/fraction, params)

            if verbose >= 1:
                print('Exponential Decay:', params, cv)
                print('Val at ',data['Pos Y adj'].max()/fraction, ' = ', valatp5pc, 's')

                fig, ax = plt.subplots()
                ax.plot(data['Time'], data['Pos Y adj'])
                ax.plot(data['Time'], baseline[0]*data['Time']+baseline[1])
                ax.plot(data['Time'], diff)
                ax.plot(data['Time'], monoExp(data['Time'], params[0], params[1], params[2]))
                ax.scatter(valatp5pc,[data['Pos Y adj'].max()/fraction] )
                fig.legend(['Pos Y adj', 'Baseline', 'Abs Diff'])
                fig.canvas.manager.set_window_title("End Baseline Index")
        except :
            print("[Error] Error with exponential decay fit.")
            if verbose >= 1:
                fig, ax = plt.subplots()
                ax.plot(data['Time'], data['Pos Y adj'])
                ax.plot(data['Time'], baseline[0]*data['Time']+baseline[1])
                ax.plot(data['Time'], diff)
                ax.plot(data['Time'], monoExp(data['Time'], params[0], params[1], params[2]))
                ax.scatter(valatp5pc,[data['Pos Y adj'].max()/fraction] )
                fig.legend(['Pos Y adj', 'Baseline', 'Abs Diff'])
                fig.canvas.manager.set_window_title("End Baseline Index")
            valatp5pc = np.nan
        finally:
            return valatp5pc
            
    def slice_data(self, trail=2, vs_baseline=False, verbose=False):
        """
        Slices off zero value trail to keep only signal.
        """

        if self.sliced:
            return True
        if not 'Pos Y adj' in self.path.columns:
            return False
        
        self.original_data = self.path.copy(deep=True)

        slice_size = int(len(self.path)/10)

        parms = linreg(self.path['Time'][:-2*slice_size], self.path['Pos Y adj'][:-2*slice_size])
        #print("Baseline regression", parms)

        if vs_baseline:
            # Calculate the baseline mean and deviation
            #bl_snr = self.signaltonoise(self.path['Pos Y adj'][:-2*slice_size])
            bl_std = self.path['Pos Y adj'].std()
            bl_avg =  self.path['Pos Y adj'].mean()
        else:
            bl_std = 20
            bl_avg = 0

        count = []
        drop = []
        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.scatter(self.path['Time'], self.path['Pos Y adj'])
            ax.fill_between(self.path['Time'], np.ones(len(self.path))*bl_avg-bl_std, np.ones(len(self.path))*bl_avg+bl_std)

        # Compare baseline mean and stdev to specific slice
        for i in np.arange(0, len(self.path), slice_size):
            temp = self.path[i:i+slice_size]
            avg = temp['Pos Y adj'].mean() 
            std = temp['Pos Y adj'].std()
            if std < 2*bl_std and abs(abs(avg) - abs(bl_avg)) < 3*bl_std:
                count.append(i)
                #print("add", i, 'to count list')
                if len(count) > trail:
                    drop.append(i)
                    #print("add", i , "to drop list")
            if verbose >= 1:
                ax.fill_between(temp['Time'], np.ones(len(temp))*avg-std, np.ones(len(temp))*avg+std)
                ax.scatter(temp['Time'], temp['Pos Y adj'])
                
                fig.canvas.manager.set_window_title("Slice Data")
                #print("Data Slices: ", count, drop)
        
        # slice off data beyond 'tail' blocks
        if len(drop) > 0:
            self.path = self.path.iloc[:min(drop)]
            #print("data length sliced", len(self.path))
        self.sliced=True
        return True

    def select_active_object(self, object_id):
        self.path = deepcopy(self.all_paths.loc[self.all_paths['object_id'] == object_id])
        self.active_object = object_id

    def processData(self, object_id=None, flatten_window=True, verbose=0, detail=False):
        """
        Normalizes base coordinates, generates smoothed curves, and calculates
        velocity and acceleration, also with smoothing

        :param time: time values for each point, by default uses\
        self.df['Time']
        :type time: list of floats, optional
        :param position: position coordinate values, by default uses\
        self.df['Position'].
        :type position: list or list of lists
        :param verbose: flag for debugging information or detailed process\
        information
        :type verbose: boolean, optional
        """
        if object_id is None and self.active_object is None:
            self.select_active_object(self.all_paths['object_id'].unique[0])

        # Prep data to expected format/naming
        self.format_data()

        # Flip the data to expected orientation and flatten bottom
        self.path['Pos X adj'] = self.rectify_and_baseline(self.path['Pos X'], type='x')
        with warnings.catch_warnings(record=True) as w:
            self.path['Pos Y adj'] = self.rectify_and_baseline(
                self.path['Pos Y'], force_flip=self.force_flip, baseline=self.end_baseline,
                type='y', verbose=verbose-1)
            if len(w) > 0:
                self.path_warning = True
        
        # Slice a long trail of zero values to keep only signal
        if self.autoslice:
            self.slice_data(verbose=verbose-1)

        # Smooth the data on the trailing edge
        if flatten_window:
            self.flatten_tail(verbose = verbose-1)

        # Smooth the rest of the data to allow minima and maxima to be found
        self.smooth_data(verbose=verbose-1)

        return self.path

    def flatten_tail(self, verbose=False):
        """
        Scale down data after main signal to reduce influence of noise.
        Avoids sudden sharp drop from just setting tail to zero.
        """
        def logistic(x, L, k, x0):
            with warnings.catch_warnings() as w:
                #print("logistic")
                val = L/(1+np.exp(-k*(-x+x0)))
                if w is not None:
                    print("[Warning] DataProcessor.flatten_tail ", w)
                return val

        def window_function(x, y, threshold, x_offset):
            if x < threshold:
                return y
            return logistic(x, 1, 0, x_offset)

        thresh_time = self.get_baseline_index(self.path)
        if thresh_time is None or np.isnan(thresh_time):
            thresh_time = self.path['Time'].max()

        window = np.ones(len(self.path))
        
        window = logistic(self.path['Time'], 1, (self.path['Time'].max()-self.path['Time'].min())/thresh_time*2, thresh_time)
        self.path['Pos Y adj'] = window*self.path['Pos Y adj']
        #print(window)

        if verbose >= 1:
            print("Cutoff at", thresh_time, 's')
            fig1, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            #ax1.plot(self.path['Time'], self.path['Pos Y adj'])
            
            ax2.plot(self.path['Time'], window)
            ax1.plot(self.path['Time'], window*self.path['Pos Y adj'])
            fig1.canvas.manager.set_window_title("Process Data: Window function")

        #print("Cutoff at", thresh_time, 's')
        #self.data = self.data.loc[self.data['Time'] < thresh_time]
        #print(self.data)

        self.smooth_data(verbose=verbose-1)

    def getBounces(self, bounce_baseline=False, verbose=0, chart_folder=None, max_bounces=10,
                   adjust_to_raw=False):
        """
        Given position, velocity, and acceleration in the y plane, returns
        the indices of bounces. Bounces correspond to local minima in Y
        Position, and Y accel, X accel, and maxima in Y accel. Bounce
        correspond to Minima in y position.

        :param use_pos: Whether or not to use position data
        :type use_pos: boolean, optional
        :param use_vel: Whether or not to use velocity data
        :type use_vel: boolean, optional
        :param use_acc: Whether or not to use acceleration data
        :type use_acc: boolean, optional
        :param verbose: flag for debugging information or detailed process\
        information
        :type verbose: boolean, optional
        """
        self.bounce_py = self.bounce_vy = self.bounce_ay = []
        
        if verbose >= 1:
            print("Writing path data to temp.csv...")
            try:
                self.path.to_csv('temp.csv')
            except:
                warnings.warn("Failed to export temporary .csv - temp.csv file open?", RuntimeWarning)
                
        self.y_range = self.path['Pos Y adj'].max()-self.path['Pos Y adj'].min()
        self.bouncemin = self.y_range*0.05+self.path['Pos Y adj'].min()

        if verbose >= 1:
            print(self.path.head())

        # Invert and see peaks (minima)
        if self.bounce_criteria['position']:
            # default distance self.framerate/5
            (self.bounce_py, _) = find_peaks(
                [-i for i in self.path['Pos Y smooth']],
                #distance = self.bouncedist,
                #threshold = self.bouncemin,
                prominence=0.6,
                #width = self.framerate/20
                )

            # Remove bounces away from ground minimum, default y_range/5
            print("Bounces in Position:",self.bounce_py)
            self.bounce_py = list(
                filter(
                    lambda i:
                    self.path['Pos Y smooth'].iloc[i] < self.bouncemax,
                    self.bounce_py))
            
            if verbose >= 1:
                print(self.bounce_py)

            if adjust_to_raw:
            # Adjust bounces to non-smoothed version
                for i in range(len(self.bounce_py)):
                    best=0
                    try:
                        for j in range(-3, 3):
                            if self.path['Pos Y adj'].iloc[min(max(0, self.bounce_py[i]+j), len(self.path))] < \
                                    self.path['Pos Y adj'].iloc[self.bounce_py[i]+best]:
                                best = j
                        self.bounce_py[i] += j
                    except IndexError as e:
                        print("IndexError in getBounces")

        # Bounces due to zero crossings in vy
        if self.bounce_criteria['velocity']:
            self.bounce_vy = [i[0] for i in np.argwhere(np.diff(
                np.sign(self.path['Vel Y smooth'])) > 0)]

        if self.bounce_criteria['acceleration']:
            # Bounces at peak in aY
            (self.bounce_ay, _) = find_peaks(self.path['Acc Y smooth'],
                                        distance=self.framerate/2)
        self.aggregate_bounces(self.bounce_py, self.bounce_vy, self.bounce_ay)

        # Remove and add bounces manually
        for i in self.bounceban:
            try:
                self.features.bounces.remove(i)
            except:
                pass
        
        if len(self.bounceforce) > 0:
            self.features.bounces += self.bounceforce
        self.features.bounces = list(tuple(self.features.bounces))
        self.features.bounces.sort()

        if bounce_baseline:            
            self.bounce_baseline_adjustment()
        else:
            self.quad_bounce_baseline()
            self.path['Pos Y norm'] = self.path['Pos Y adj']
        
        self.features.bounces = self.filter_by_noise(self.features.bounces)

        if verbose or chart_folder is not None:
            print("Plotting bounces")
            self.plot_bounces(chart_folder=chart_folder, verbose=verbose-1)

        if len(self.features.bounces) > max_bounces:
            print("Bounce detection excessive, indicates only noise from non-moving objects is present.")
            self.features.bounces=[]
            #raise ValueError(f"Too many bounces in {self.pathfile}, probably noise, try again later.")
        return self.features.bounces

    def get_bounces_cwt(self, verbose=0):
        """
        Get bounces using wavelet transformation instead of standard
        """
        bounces = find_peaks_cwt(self.path['Pos Y adj'], np.arange(self.framerate/5,self.framerate/2), min_snr=0.5, noise_perc=10, window_size=10)
        if verbose >= 1:
            print("CWT bounces", self.path['Time'].iloc[bounces])

        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.plot(self.path['Time'], self.path['Pos Y adj'])
            ax.scatter(self.path['Time'].iloc[bounces], self.path['Pos Y adj'].iloc[bounces], color='red')

        return bounces

    def filter_by_noise(self, points, column='Pos Y adj'):
        """
        Remove bounces too low within noise level
        """
        #print("Checking noise level of areas")
        step = self.framerate/4
        temp_points = []
        for i in points:
            # slice nearby data
            lower = int(i-step) if i-step > 0 else 0
            upper = int(i+step) if i+step < len(self.path) else len(self.path)
            values = self.path[column].iloc[lower:upper].values.tolist()
            # Compare Z value to noise level
            meanval = np.mean(values)
            stdval = np.std(values)
            if stdval == 0:
                snr=1000
            else:
                snr = meanval/stdval
            if (snr < self.snr_thresh) and (meanval < self.y_range*self.snr_filter_thresh):
                pass
                #print("Critical point ", i, " mean, stdev (", meanval, "+/-", stdval, ")")
            else:
                temp_points += [i]
        
        return temp_points

    def bounce_baseline_adjustment(self):
        """
        Adjust baseline to consider all bounces as zero height.
        Corrects for perspective and lens distortion.
        """
        # Calculate baseline
        ind = self.features.bounces
        new_baseline = [0]*len(self.path)
        self.baselines = [0]*len(self.path)

        if len(ind) <= 1:
            self.path['Pos Y norm'] = self.path['Pos Y adj']
            return
        
        a, b, c, d, e = stats.linregress(
            [ind[0], ind[1]],
            y=[self.path['Pos Y adj'].iloc[ind[0]],
            self.path['Pos Y adj'].iloc[ind[1]]])
    
        for j in range(0, ind[1]):
            self.baselines[j] = a*j+b
            new_baseline[j
            ] = self.path['Pos Y adj'].iloc[j] - \
                self.baselines[j]

        for i in range(1, len(ind)):
            a, b, c, d, e = stats.linregress(
                [ind[i-1], ind[i]],
                y=[self.path['Pos Y adj'].iloc[ind[i-1]],
                self.path['Pos Y adj'].iloc[ind[i]]])

            for j in range(ind[i-1], ind[i]):
                self.baselines[j] = a*j+b
                new_baseline[j] = self.path['Pos Y adj'].iloc[j] - self.baselines[j]

        for j in range(ind[-1], len(self.path)-1):
            self.baselines[j] = a*j+b
            new_baseline[j] = self.path['Pos Y adj'].iloc[j] - \
                self.baselines[j]

        #print(baseline)
        self.path['Pos Y norm'] = new_baseline
        self.path['Pos Y bl'] = self.baselines

    def quad_bounce_baseline(self, verbose=False):
        """
        Use quadratic function as bounce baseline instead of linear
        """
        # Calculate baseline
        if len(self.features.bounces) < 1:
            self.path['Pos Y norm'] = self.path['Pos Y adj']
            return	
        if len(self.features.bounces) == 1:
            self.path['Pos Y norm'] = self.path['Pos Y adj'] - self.path['Pos Y adj'].iloc[self.features.bounces[0]]
            return

        def monoExp(x, m, t):
            with warnings.catch_warnings() as w:
                #print("quad_bounce_baseline monoExp")
                val =  m * np.exp(-t * x)
                if w is not None:
                    print("[Warning] DataProcessor.quad_bounce_baseline ", w)
            return val
        
        ind = self.features.bounces
        try:    
            p0 = (0.1, 0.1) # start with values near those we expect
            params, cv = curve_fit(
                monoExp,
                self.path['Time'].iloc[ind],
                self.path['Pos Y adj'].iloc[ind],
                p0,
                bounds=((0, 0), (np.inf, np.inf)))
            self.path['Pos Y bl'] = monoExp(self.path['Time'], *params)
        except Exception as e:
            warnings.warn("Error fitting exponential baseline", RuntimeWarning)
            a,b,c = np.polyfit(self.path['Time'].iloc[ind], self.path['Pos Y adj'].iloc[ind], 2)
            #zeros = 
            self.path['Pos Y bl'] = self.path['Time'].apply(lambda x: a*x**2+b*x+c if x <= -b/(2*a) else 0)
        
        self.path['Pos Y norm'] = self.path['Pos Y adj'] - self.path['Pos Y bl']

        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.plot(self.path['Time'], self.path['Pos Y norm'])
            ax.plot(self.path['Time'], self.path['Pos Y adj'])
            ax.plot(self.path['Time'], self.path['Pos Y bl'])
            ax.legend(['Normalized', 'Raw', 'Baseline'])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Position (px)')
            fig.canvas.manager.set_window_title('After-Bounce Baseline')

    def aggregate_bounces(self, bounce_py, bounce_vy, bounce_ay):
        """
        Combine bounces detected from position, velocity, and acceleration
        to remove any overlap, and adjust for precision.
        """
        bounce_index = []
        if int(self.bounce_criteria['position']) + \
                int(self.bounce_criteria['velocity']) + \
                int(self.bounce_criteria['acceleration']) > 1:
            if self.bounce_criteria['position']:
                for i in self.bounce_py:
                    if self.bounce_criteria['velocity']:
                        for j in self.bounce_vy:
                            if abs(i - j) < self.framerate/3:
                                bounce_index.append((i, j-1))
                    if self.bounce_criteria['acceleration']:
                        for j in self.bounce_ay:
                            if abs(i - j) < self.framerate/3:
                                bounce_index.append((i, j-2))
                    if not self.bounce_criteria['velocity'] and \
                            not self.bounce_criteria['acceleration']:
                        bounce_index.append((i, i))
            else:
                if self.bounce_criteria['velocity']:# and use_acc:
                    for i in bounce_vy:
                        for j in bounce_ay:
                            if abs(i - j) < self.framerate/6:
                                bounce_index.append((i-1, j-2))

            bounce_index.sort()

            final_list = list(set([i[0] for i in bounce_index] +
                                  [i[1] for i in bounce_index]))
            final_list.sort()

            self.features.bounces = []
            for k, g in groupby(final_list,
                                lambda x: round(x/self.bouncewindow)):
                grp = list(g)
                if(len(grp) > 0):
                    grp_avg = int(np.ceil(sum(grp)/len(grp)))+1
                    self.features.bounces.append(grp_avg)
        else:
            if int(self.bounce_criteria['position']):
                self.features.bounces = self.bounce_py
            elif int(self.bounce_criteria['velocity']):
                self.features.bounces = self.bounce_vy
            elif int(self.bounce_criteria['acceleration']):
                self.features.bounces = self.bounce_ay

    def plot_bounces(self, chart_folder=None, verbose=False):
        """
        Generate charts to display bounce detection results
        """
        charts = 0
        if self.bounce_criteria['position']:
            charts += 1
        if self.bounce_criteria['velocity']:
            charts += 1
        if self.bounce_criteria['acceleration']:
            charts += 1

        f, axes = plt.subplots(1, charts)
        try:
            axes[0]
        except:
            axes = [axes]
        chart = 0
        if self.bounce_criteria['position']:
            print("preparing position chart")
            #axes[chart].plot(self.path['Time'], self.path['Pos Y adj'])
            if 'Pos Y norm' in self.path.columns:
                axes[chart].plot(self.path['Time'], self.path['Pos Y norm'],
                    picker=10)
            axes[chart].plot(self.path['Time'], self.path['Pos Y smooth'],
                picker=10)
            if hasattr(self, 'baselines'):
                axes[chart].plot(self.path['Time'], self.baselines)

            if 'Pos Y norm' in self.path.columns:
                axes[chart].scatter(
                    [self.path['Time'].iloc[i] for i in self.features.bounces],
                    [self.path['Pos Y norm'].iloc[i] for i in self.features.bounces],
                    picker=True)
            axes[chart].scatter(
                [self.path['Time'].iloc[i] for i in self.features.bounces],
                [self.path['Pos Y smooth'].iloc[i] for i in self.features.bounces],
                picker=True)
            axes[chart].legend(['Normalized', 'Smooth',
                                'Baseline', 'Bounces', 'Bounce PY'])
            chart += 1
            pd.set_option("display.max_rows", None,
                            "display.max_columns", None)
            #print(self.path[['Pos Y norm', 'Pos Y smooth',
            #                    'Pos Y norm', 'Pos Y smooth']])

        if self.bounce_criteria['velocity']:
            print("preparing velocity chart")
            axes[chart].plot(self.path['Time'], self.path['Vel Y smooth'])
            axes[chart].scatter(
                [self.path['Time'].iloc[i] for i in self.bounce_vy],
                [self.path['Vel Y smooth'].iloc[i] for i in self.bounce_vy],
                picker=True)
            if verbose >= 1:
                print("Bounces py:", self.bounce_py,
                        [(self.path['Time'].iloc[i],
                        self.path['Pos Y smooth'].iloc[i])
                        for i in self.bounce_py])
                print("Bounces vy:", self.bounce_vy,
                        [(self.path['Time'].iloc[i],
                        self.path['Vel Y'].iloc[i]) for i in self.bounce_vy])
            chart += 1

        if self.bounce_criteria['acceleration']:
            print("preparing acceleration chart")
            axes[chart].plot(self.path['Time'].iloc, self.acc_smooth[1],
                [self.path['Time'].iloc[i] for i in self.bounce_ay],
                [self.path['Acc Y smooth'].iloc[i] for i in self.bounce_ay],
                picker=True)
            if verbose >= 1:
                print("Bounces aY:", self.bounce_ay,
                    [(self.path['Time'].iloc[i],
                    self.path['Acc Y'].iloc[i]) for i in self.bounce_ay])

        if verbose >= 1:
            print("Agglomerated bounces:", self.features.bounces)
            print("Final Bounces: ", self.features.bounces)
        f.canvas.manager.set_window_title("Bounce Detection")
        
        self.add_handlers("bounces", f)
        
        if chart_folder is not None:
            file = chart_folder+os.path.basename(self.pathfile).split('.')[0]+'_bounces.png'
            try:
                plt.savefig(file)
                print("Wrote Bounce chart to file: ", os.path.abspath(file))
            except:
                print(["[Error] exporting chart failed"])
        if verbose >= 1:
            plt.show()
        #plt.clf()

    def getCrests(self, chart_folder=None, verbose=False):
        """
        Gets the apex correspond to maxima in y position
        TODO: Only look for maxima during parabola/freefall moments

        :param verbose: flag for debugging information or detailed process\
        information
        :type verbose: boolean, optional
        """

        (crest_py, _) = find_peaks(self.path['Pos Y norm'],
                                   distance=self.crestdist,
                                   #threshold=self.crestmin
                                   prominence=0.6,
                                   width=[self.framerate/20, self.framerate/2]
                                   )
        if verbose >= 1:
            print("Raw crest list", self.path['Time'].iloc[crest_py])

        # Adjust bounces to non-smoothed version
        for i in range(len(crest_py)):
            try:
                for j in range(-3, 3):
                    if self.path['Pos Y norm'].iloc[max(0, crest_py[i]+j)] > \
                            self.path['Pos Y norm'].iloc[crest_py[i]]:
                        crest_py[i] += j
            except IndexError:
                pass
                #No idea why this is sometimes out of bounds...

        # Crests due to zero crossings in vy
        crest_vy = [i[0] for i in
                    np.argwhere(np.diff(
                        np.sign(self.path['Vel Y smooth'].dropna())) < 0)]

        crest_index = []
        for i in crest_py:
            for j in crest_vy:
                if abs(i - j) < self.framerate/6:
                    #bounce_index.append( (i, j-1, (ptime[i], y[i])) )
                    crest_index.append(i)

        crests = []
        for k, g in groupby(crest_index, lambda x: round(x/self.crestwindow)):
            grp = list(g)
            if(len(grp) > 0):
                grp_avg = int(round(sum(grp)/len(grp)))
                crests.append(grp_avg)    # Store group iterator as a list

        self.features.crests = crest_py
        if verbose >= 1:
            print("Crests detected from position: ", self.features.crests)
            print("Filtering noisy peaks")

        self.features.crests = self.filter_by_noise(self.features.crests)
        
        # Remove and add bounces manually
        for i in self.crestban:
            try:
                self.features.crests.remove(i)
            except:
                pass

        self.cors = []
        for i in range(1, len(self.features.crests)):
            self.cors.append(
                self.path['Pos Y adj'].iloc[self.features.crests[i]]/
                self.path['Pos Y adj'].iloc[self.features.crests[i-1]])
        
        if verbose or chart_folder is not None:
            self.plot_crests(chart_folder, verbose=verbose)

        if verbose >= 1:
            print("Coefficients of Restitution:", self.cors)
            print("Crest ymax: ", crest_py)
            print("Crest vy: ", crest_vy)
            print("Crest index: ", crest_index)
            #print("Final crests", crests,
            #      [self.path['Time'][i] for i in crests])
            
        return ('Crests', crests)
    
    def plot_crests(self, chart_folder=None, verbose=False):
        """
        Generate charts to display crest detection results
        """

        f, ax = plt.subplots(1, 1)
        ax.plot(self.path['Time'], self.path['Pos Y smooth'])
        ax.plot(self.path['Time'], self.path['Pos Y adj'])
        ax.plot(self.path['Time'], self.path['Pos Y norm'])
        ax.scatter([self.path['Time'].iloc[i] for i in self.features.crests],
                [self.path['Pos Y adj'].iloc[i] for i in self.features.crests])
        ax.legend(['Smooth', 'Adj', 'Norm', 'Crests'])

        f.canvas.manager.set_window_title("Crest Detection")
        self.add_handlers("crests", f)
        
        if chart_folder is not None:
            file = chart_folder+os.path.basename(self.pathfile).split('.')[0]+'_crests.png'
            if not os.path.exists(os.path.basename(file)) and not os.path.exists(os.path.dirname(file)):
                os.mkdir(os.path.dirname(file))
            plt.savefig(file)
        if verbose >= 1:
            plt.show()
        #plt.clf()

    def calc_speed_linear_regression(self, data, verbose=0):
        """
        Calculate speed as tangent from data near point
        """
        data = data[['Time', 'Pos X', 'Pos Y']].dropna()
        x_fit = stats.linregress(data['Time'], data['Pos X'])
        x_vel = x_fit[0] / self.features.px_conv
        y_fit = stats.linregress(data['Time'], data['Pos Y'])
        y_vel = y_fit[0] / self.features.px_conv
        
        if verbose >= 1:
            fig, ax = plt.subplots(2)
            ax[0].scatter(self.path['Time'], self.path['Pos X'])
            ax[0].scatter(data['Time'], data['Pos X'])
            ax[0].plot(data['Time'], data['Time']*x_fit[0]+x_fit[1])
            ax[1].scatter(self.path['Time'], self.path['Pos Y'])
            ax[1].scatter(data['Time'], data['Pos Y'])
            ax[1].plot(data['Time'], data['Time']*y_fit[0]+y_fit[1])
            plt.show()
        return np.abs([x_vel, y_vel])

    def get_window(self, mid, avoid, width, win_type, forceavoid=False):
        """
        Slice data to get points before and after a point
        to avoid noisy data at contact point.
        """
        if win_type == 'before':
            low = mid - avoid - width
            hi = mid - avoid
        elif win_type == 'after':
            low = mid + avoid
            hi = mid + avoid + width

        low = low if low > 0 else 0
        if hi >= len(self.path):
            hi = len(self.path)-1

        if win_type == 'before':
            if hi-low <= 2 and not forceavoid:
                low = mid - width - 2
        else:
            if hi-low <= 2 and not forceavoid:
                hi = mid + width + 2
        
        low = low if low > 0 else 0
        if hi >= len(self.path):
            hi = len(self.path)-1

        if verbose >= 1:
            print("Before point - Low index:", low, " high index: ", hi)
        if hi-low < 2:
            return (None, (low, hi))
        data = self.path.iloc[low:hi]
        
        return (data, (low, hi))

    def get_speeds_before_and_after(self, index, window=5, avoid=1, verbose=0, forceavoid=False):
        """
        Get average speeds before and after a point

        :param index: index of reference point
        :type index: integer
        :param window: width of window in which to measure average speed
        :type window: integer
        :param avoid: points not considered before and after the target index
        :type avoid: integer
        :param verbose: flag for debugging information or detailed process\
        information
        :type verbose: boolean, optional

        returns tuple of two tuples (((inspeed x, inspeed y) ,(outspeed x, outspeed y)), (inangle, outangle))
        """
        if verbose >= 1:
            fig, ax = plt.subplots()
            ax.plot('Time', 'Pos Y', data=self.path)
        # Before point
        try:
            data, (low, hi) = self.get_window(index, avoid, window, 'before', forceavoid)
            if data is None:
                raise ValueError("Window too small")
            if verbose >= 1:
                print("Before point - Low index:", low, " high index: ", hi)
           
            inspeed = self.calc_speed_linear_regression(data, verbose=verbose-1)
            inangle = np.arctan(inspeed[1]/inspeed[0])/np.pi*180 if abs(inspeed[0]) > 0 else 90

            if verbose >= 1:
                print("Before index:", index, "Window:", low, hi)
                print("window: ", window, "px_conv", self.features.px_conv)
                print('X vel:', inspeed[0], 'Y vel:', inspeed[1])
                print('Angle in:', inangle)

                ax.plot([])
        except Exception as e:
            warnings.warn("In speed calculation error: "+ str(e), RuntimeWarning)
            inspeed = (None, None)
            inangle = None
      
        try:
            data, (low, hi) = self.get_window(index, avoid, window, 'after', forceavoid)
            if verbose >= 1:
                print("After point - Low index:", low, " high index: ", hi)
           
            outspeed = self.calc_speed_linear_regression(data, verbose=verbose-1)
            outangle = np.arctan(outspeed[1]/outspeed[0])/np.pi*180 if abs(outspeed[0]) > 0 else 90
    
            if verbose >= 1:
                print("After index:", index, "Window:", low, hi)
                print("window: ", window, "px_conv", self.features.px_conv)
                #print(x_fit, y_fit)
                print('X vel:', outspeed[0], 'Y vel:', outspeed[1])
                print('Angle out:', outangle)
        except Exception as e:
            outspeed = (None, None)
            outangle = None
            warnings.warn("Out speed calculation error: "+ str(e), RuntimeWarning)

        return {'speed':{'in':inspeed, 'out':outspeed}, 'angle':{'in':inangle, 'out':outangle}}

    def get_speeds(self, verbose=False):
        """
        Calculate speeds before and after surface bounces
        """

        self.features.speeds = []
        for i in self.features.bounces:
            if verbose >= 1:
                print("Calculating for bounce" , i)
            self.features.speeds += [self.get_speeds_before_and_after(i, verbose=verbose)]
        
        if verbose >= 1:
            fig, ax = plt.subplots(3)
            X = np.arange(len(self.features.speeds))
            in_x = [speed['speed']['in'][0] for speed in self.features.speeds]
            in_y = [speed['speed']['in'][1] for speed in self.features.speeds]
            #print(X, y)
            ax[0].bar(X, in_x)
            ax[0].bar(X+0.25, in_y)
            
            out_x = [speed['speed']['out'][0] for speed in self.features.speeds]
            out_y = [speed['speed']['out'][1] for speed in self.features.speeds]
            ax[1].bar(X, out_x)
            ax[1].bar(X+0.25, out_y)

            ax[2].plot(self.path['Time'], self.path['Pos Y norm'])
            #for spd in self.features.speeds:
            #    spd['speed']['']
            
            plt.show()
        return self.features.speeds

    def fit_to_curves(self, verbose=False):
        """
        Curvefit x, y with parabola between bounces
        use values to calculate gravity
        """
        self.features.pos_fit = []

        if len(self.features.crests) > 0 and len(self.features.bounces) > 0 and \
        self.features.crests[0] > 5 and self.features.crests[0] < self.features.bounces[0]:
            first = self.features.crests[0]-5
            if first < 0:
                first = 0
        else:
            first = 0
        if verbose >= 1:
            print(first, self.features.bounces, len(self.path['Time'])-1)
        indices = sorted([first] + self.features.bounces + [len(self.path['Time'])-1])
        self.indices = indices

        self.features.fit_bounces = []
        self.features.fit_crests = []
        px_conv = []

        if verbose >= 1:
            print("Indices:", indices)

        # Curve fit Y position
        for i in range(1, len(indices)):
            #if (indices[i]-indices[i-1] < framerate):
            #   continue
            try:
                with warnings.catch_warnings() as w:
                    fit = np.polyfit(
                        self.path['Time'].iloc[indices[i-1]: indices[i]],
                        self.path['Pos Y norm'].iloc[indices[i-1]: indices[i]],
                        2)
                    if not w is None:
                        print("[WARNING]", w)
                self.features.pos_fit.append(fit)
                # Get crests from curve-fit
                a, b, c = self.features.pos_fit[i-1]
                xval = -b / (2*a)
                # Pixel to meter conversion
                px_conv.append(abs(a/(0.5*9.81)))
            except Exception as err:
                print("[Error]", err)
                a,b,c = [0, 0, 0]
                xval = 0

            self.features.fit_crests.append((xval, a*xval**2 + b*xval + c))
            #print ("Curve fit", i, ":", pos_fit[i-1])

        # Use gravity as metric to determine scale
        # average pixel to distance conversion (px /m)
        #self.features.px_conv = sum(px_conv)/len(px_conv)
        if px_conv==[]:
            px_conv = abs(fit[0]/(0.5*9.81))
        else:
            px_conv = np.median(px_conv)
        
        if hasattr(self, 'scale'):
            self.features.px_conv = self.scale
            self.calc_pxconv = px_conv
        else:
            self.features.px_conv = self.calc_pxconv =  px_conv

        print("Pixel scale: ", self.scale if hasattr(self, "scale") else "-", "Calculated scale from fit: ", self.calc_pxconv)

        # Solve for function meeting points from fit, e.g. bounces
        for i in range(0, len(self.features.pos_fit)-1):
            def f(x):
                def f1(x):
                    return self.features.pos_fit[i][0]*x**2 + self.features.pos_fit[i][1]*x + \
                        self.features.pos_fit[i][2]
                def f2(x):
                    return self.features.pos_fit[i+1][0]*x**2 + self.features.pos_fit[i+1][1]*x \
                        + self.features.pos_fit[i+1][2]
                return f1(x)-f2(x)

            xval = fsolve(f, self.path['Time'].iloc[self.features.bounces[i]])[0]
            yval = self.features.pos_fit[i][0]*xval**2 + self.features.pos_fit[i][1]*xval + \
                self.features.pos_fit[i][2]
            self.features.fit_bounces.append((xval/self.features.px_conv, yval/self.features.px_conv))

        if verbose >= 1:
            f, ax = plt.subplots(1, 1)
            ax.plot(self.path['Time'],
                    [i/self.features.px_conv for i in self.path['Pos Y norm']])
        # Full fit curve
        fitcurves = [[]]*2
        self.features.fit_speeds = []
        # ((inspeed, outspeed), (inangle, outangle))
        for i in range(0, len(self.features.pos_fit)):
            def slope(t):
                # Derivative - i.e. velocity from fit
                return 2*self.features.pos_fit[i][0]*t + self.features.pos_fit[i][1]
            
            # calculate fit values for whole trace
            fit = [self.features.pos_fit[i][0]*j**2 + self.features.pos_fit[i][1]*j +
                   self.features.pos_fit[i][2] for j in
                   self.path['Time'].iloc[self.indices[i]:self.indices[i+1]]]
            
            # Y speed in, Y speed out
            fit_speeds = np.abs([slope(self.indices[i])/(self.features.px_conv**2),
                                 slope(self.indices[i+1])/(self.features.px_conv**2)])
            self.features.fit_speeds = self.features.fit_speeds + [fit_speeds]

            if verbose >= 1:
                # Plot each curve fit separately
                print("Fit Speeds:!",self.features.fit_speeds)
                ax.plot(
                    self.path['Time'].iloc[self.indices[i]:self.indices[i+1]],
                    fit/self.features.px_conv)

                print(fitcurves[0],
                    self.path['Time'].iloc[self.indices[i]:self.indices[i+1]])
            fitcurves[0] = fitcurves[0] + \
                self.path['Time'].iloc[self.indices[i]:self.indices[i+1]].tolist()
            fitcurves[1] = fitcurves[1] + fit

            # Calculate Speed values
            #low = self.indices[i] - 5 if self.indices[i] >= 5 else 0
            #xspeedin = linreg(self.path['Time'].iloc[low:self.indices[i]],
            #                  self.path['Pos X adj'].iloc[low:self.indices[i]])
            #inspeed = np.sqrt(slope(self.indices[i])**2 + xspeedin.slope**2)

        if(verbose):
            print("Fit Bounces:", self.features.fit_bounces)
            print("Fit Crests: ", self.features.fit_crests)
            print("Framerate: ", self.framerate)
            print("Unit Conversion (px/m):", self.features.px_conv, '\n', px_conv)
            for i in self.features.pos_fit:
                print(str(i[0]) + '*t2 + ' + str(i[1]) + '*t + ' + str(i[2]))
            ax.scatter(
                [self.path['Time'].iloc[i] for i in self.features.bounces],
                [self.path['Pos Y norm'].iloc[i]/self.features.px_conv for i in self.features.bounces])
            plt.show()

    def show_charts(self):
        """
        Plot charts
        """
        # Create three subplots

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)

        # Position
        ax1.set_title('Position')
        #for i in self.pos:
        #    ax1.plot(self.time[0], i/self.features.px_conv)
        #for i in self.pos_smooth:
        #    ax1.plot(self.time[0], i/self.features.px_conv)

        self.fitcurves = [[]]*2
        for i in range(0, len(self.features.pos_fit)):
            fit = [self.features.pos_fit[i][0]*j**2 + self.features.pos_fit[i][1]*j +
                   self.features.pos_fit[i][2] for j in
                   self.path['Time'].iloc[self.indices[i]:self.indices[i+1]]]
            ax1.plot(self.path['Time'].iloc[self.indices[i]:self.indices[i+1]],
                     fit/self.features.px_conv)
            ax1.plot(self.path['Time'], self.path['Pos Y norm'])
            if verbose >= 1:
                print(self.fitcurves[0],
                      self.path['Time'].iloc[self.indices[i]:self.indices[i+1]])
            self.fitcurves[0] = self.fitcurves[0] + \
                self.path['Time'].iloc[self.indices[i]:self.indices[i+1]].tolist()
            self.fitcurves[1] = self.fitcurves[1] + fit

        ax1.scatter([self.path['Time'].iloc[i] for i in self.features.bounces],
                    [self.path['Pos Y norm'].iloc[i]/self.features.px_conv
                     for i in self.features.bounces])

        ax1.scatter([self.path['Time'].iloc[i] for i in self.features.crests],
                    [self.path['Pos Y norm'][i]/self.features.px_conv
                     for i in self.features.crests])

        ax1.legend(['X', 'Y', 'Bounces', 'Crests'])

        # Velocity
        ax2.set_title("Velocity")
        ax2.plot(self.path['Time'], self.path['Vel Y smooth']/(self.features.px_conv/2))
        ax2.plot(self.path['Time'], self.path['Vel Y']/(self.features.px_conv/2))
        ax2.scatter([self.path['Time'].iloc[i] for i in self.features.bounces],
                    [self.path['Vel Y smooth'].iloc[i]/(self.features.px_conv/2)
                     for i in self.features.bounces])
        ax2.scatter([self.path['Time'].iloc[i] for i in self.features.crests],
                    [self.path['Vel Y smooth'].iloc[i]/(self.features.px_conv/2)
                     for i in self.features.crests])
        ax1.legend(['X', 'Y', 'Bounces', 'Crests'])

        # Acceleration
        ax3.set_title("Acceleration")
        ax3.plot(self.path['Time'], self.path['Acc X smooth'])
        ax3.plot(self.path['Time'], self.path['Acc Y smooth'])
        ax3.scatter([self.path['Time'].iloc[i] for i in self.features.bounces],
                    [self.path['Acc Y smooth'].iloc[i] for i in self.features.bounces])
        ax3.legend(['X', 'Y', 'Bounces'])

        f.canvas.set_window_title("Ball Position")

        plt.show()

    def write_csv_stats(self, *args, **kwargs):
        # Old function, renamed for simplicity
        warnings.warn("Deprecated, use write_stats_to_csv instead", DeprecationWarning)
        self.write_stats_to_csv(*args, **kwargs)
        
    def build_result_obj(self, filename=None):
        """
        Build object with all the desired metrics/results
        """
        csvobj = {}

        if filename is None:
            filename = self.pathfile
        
        csvobj['Filename'] = os.path.basename(filename)
        csvobj['Folder'] = os.path.dirname(filename)
        csvobj['Video File'] = os.path.basename(filename)[:8].upper()+'.MOV'
        csvobj['Analysis Date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        csvobj['Lane'] = self.features.lane if hasattr(self.features, 'lane') else None
        csvobj['Angle'] = self.features.angle if hasattr(self.features, 'angle') else None
        csvobj['Condition'] = self.features.condition if hasattr(self.features, 'condition') else None

        # Height max 1-4
        for i in range(4):
            if i < len(self.features.crests):
                csvobj['Height Max '+str(i+1)] = \
                    self.path['Pos Y norm'].iloc[self.features.crests[i]] / self.features.px_conv
            else:
                csvobj['Height Max '+str(i+1)] = None

        # Coef rest 1-3
        for i in range(1,4):
            if i >= len(self.features.crests):
                csvobj['Coef Rest '+str(i+1)] = None
                continue

            csvobj['Coef Rest '+str(i)] = \
                self.path['Pos Y norm'].iloc[self.features.crests[i]] / \
                    self.path['Pos Y norm'].iloc[self.features.crests[i-1]]

        # Bounce Position 1-3
        for i in range(3):
            if i < len(self.features.bounces):
                csvobj['Position Bounce '+str(i+1)] = \
                    self.path['Pos X adj'].iloc[self.features.bounces[i]] /\
                        self.features.px_conv
            else:
                csvobj['Position Bounce '+str(i+1)] = None
        
        # Angle in, angle out 1-4
        #((inspeed, outspeed), (inangle, outangle))
        for i in range(4):
            if (i >= len(self.features.speeds)) or (not 'angle' in self.features.speeds[i]):
                csvobj['Angle In Bounce '+str(i+1)] = None
                csvobj['Angle Out Bounce '+str(i+1)] = None
                continue

            angles = [self.features.speeds[i]['angle']['in'], self.features.speeds[i]['angle']['out']]
            if not None in angles:
                angles = [i if i > 0 else i+180 for i in angles if i is not None]
            
            csvobj['Angle In Bounce '+str(i+1)] = angles[0]
            csvobj['Angle Out Bounce '+str(i+1)] = angles[1]

        # Speed in, speed out 1-4
        for i in range(4):
            if (i >= len(self.features.speeds)) or (not 'speed' in self.features.speeds[i]):
                for k,j in [(k,j) for k in ['In ', 'Out '] for j in ['', ' X', ' Y']]:
                    csvobj['Speed '+k+str(i+1)+j] = None
                #csvobj['Speed Out '+str(i+1)] = None
                continue

            csvobj['Speed In '+str(i+1)] = self.hypothenuse(self.features.speeds[i]['speed']['in'])
            csvobj['Speed In '+str(i+1)+ " X"] = self.features.speeds[i]['speed']['in'][0]
            csvobj['Speed In '+str(i+1)+ " Y"] = self.features.speeds[i]['speed']['in'][1]
            csvobj['Speed Out '+str(i+1)] = self.hypothenuse(self.features.speeds[i]['speed']['out'])
            csvobj['Speed Out '+str(i+1)+ " X"] = self.features.speeds[i]['speed']['out'][0]
            csvobj['Speed Out '+str(i+1)+ " Y"] = self.features.speeds[i]['speed']['out'][1]

        for i in range(4):
            if (i >= len(self.features.fit_speeds)):
                csvobj['Fit Speed In Y '+str(i)] = None
                csvobj['Fit Speed Out Y '+str(i)] = None
                continue
            csvobj['Fit Speed In Y '+str(i)] = self.features.fit_speeds[i][0]
            csvobj['Fit Speed Out Y '+str(i)] = self.features.fit_speeds[i][1]


        # Scale information, and error in data processing?
        csvobj['Pixel Scale (px/m)'] = self.features.px_conv
        if hasattr(self, "calc_pxconv"):
            csvobj['Calculated Pixel Scale (px/m)'] = self.calc_pxconv
        csvobj['Path Data Warning'] = self.path_warning

        return csvobj

    def write_stats_to_csv(self, csv_fname='temp.csv', filename=None):
        csvobj = self.build_result_obj(filename)

        newfile = not os.path.exists(csv_fname)
        while True:
            try:
                with open(csv_fname, mode='a+') as csv_file:
                    data_writer = csv.writer(csv_file, delimiter='\t',
                                lineterminator='\r',
                                quoting=csv.QUOTE_MINIMAL)

                    if newfile:
                        data_writer.writerow(csvobj.keys())
                    data_writer.writerow(csvobj.values())
                    return csvobj
            except PermissionError:
                print("[ERROR] Permission error, close file and try again.")
                tval = input("Try again?")
                if tval.lower() == 'n':
                    break

    def write_csv_path(self, *args, **kwargs):
        warnings.warn("Deprecated, use write_path_to_csv instead", DeprecationWarning)
        self.write_path_to_csv(*args, **kwargs)

    def write_path_to_csv(self, csv_fname='output/temp.csv', filename='-'):
        """
        Write ball path used to CSV file.
        """
        self.df = pd.DataFrame(
            index=range(2, len(self.pos[0])),
            data={
                'Time': self.time[0][2:],
                'Pos X': self.pos[0][2:],
                'Pos Y': self.path['Pos Y norm'][2:],
                'Pos X Smooth': self.pos_smooth[0][2:],
                'Pos Y Smooth': self.pos_smooth[1][2:],
                'Pos Y Fit': self.fitcurves[1][1:],
                'Vel X': self.vel[0][1:],
                'Vel Y': self.vel[1][1:],
                'Vel X Smooth': self.vel_smooth[0][1:],
                'Vel Y Smooth': self.vel_smooth[1][1:],
                'Acc X': self.acc[0],
                'Acc Y': self.acc[1],
                'Acc X Smooth': self.acc_smooth[0],
                'Acc Y Smooth': self.acc_smooth[1]
            })
        self.df.columns = pd.MultiIndex.from_tuples([
            ('', 'Time'),
            ('Position', 'X'),
            ('Position', 'Y'),
            ('Position', 'X Smooth'),
            ('Position', 'Y Smooth'),
            ('Position', 'Y Fit'),
            ('Velocity', 'X'),
            ('Velocity', 'Y'),
            ('Velocity', 'X Smooth'),
            ('Velocity', 'Y Smooth'),
            ('Acceleration', 'X'),
            ('Acceleration', 'Y'),
            ('Acceleration', 'X Smooth'),
            ('Acceleration', 'Y Smooth')
            ])
        #print(self.df)
        self.df.to_csv(csv_fname, sep=',', line_terminator='\r')

    def analyze(self, x, y):
        """
        Fourier analysis of data
        """
        T = np.average(np.diff(x))
        N = len(x)
        yf = fft(y)
        xf = np.linspace(0, 1/(2*T), N//2)
        plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        plt.show()
    
    def add_handlers(self, name, fig, handlers=(None, None, None)):
        # Attach events to chart for clicks and keypresses

        def onclick(event):
            if event.x is None:
                print("Click off chart")
                return
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
        
        def keypress(event):
            print('you pressed', event.key, event.xdata, event.ydata)

        def onpick(event):
            if event.mouseevent.dblclick and event.mouseevent.button==3:
                self.bounceforce=[]
                self.bounceban=[]
                return
            ind = event.ind
            if not type(ind) == type(1):
                ind = ind[0]
            print("Pick index:", ind)
            try:
                line = event.artist
                xdata, ydata = line.get_data()
                x = xdata[ind]
                y = ydata[ind]
                print('on pick line:', x, y)
                if event.mouseevent.button == 1:
                    print("Line chosen x:", x, ",", y, "() click, add")
                    self.bounceforce += [int(ind)]
                
            except AttributeError as e:
                if name == "bounces":
                    # Not line, maybe scatter
                    bindex = self.features.bounces[ind]
                    x = self.path['Time'].iloc[bindex]
                    ya = self.path['Pos Y adj'].iloc[bindex]
                    yn = self.path['Pos Y norm'].iloc[bindex]
                    if event.mouseevent.button == 3:
                        print("Scatter chosen x:", x, ",", ya, "(", yn, ")right click, remove")
                        if ind in self.bounceforce:
                            self.bounceforce.remove(ind)
                        self.bounceban += [bindex]
                elif name == "crests":
                    # Not line, maybe scatter
                    crestex = self.features.crests[ind]
                    x = self.path['Time'].iloc[crestex]
                    ya = self.path['Pos Y adj'].iloc[crestex]
                    yn = self.path['Pos Y norm'].iloc[crestex]
                    if event.mouseevent.button == 3:
                        print("Scatter chosen x:", x, ",", ya, "(", yn, ")right click, remove")
                        if ind in self.crestforce:
                            self.crestforce.remove(ind)
                        self.crestban += [crestex]

        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('pick_event', onpick)
        fig.canvas.mpl_connect('key_press_event', keypress)

############

def autoprocess_folders(cones, folders, 
        output_file="../Data/auto_results_v5.csv",
        lane_data_file = "../Data/video_lane_angles.csv", verbose=0):
    """
    Used for batch processing of scripts. May not work properly now.
    """
    matplotlib.use('Agg')

    cones = pd.read_csv('../Media/Calhoun Lanes/cone_detections_summary.csv', sep='[\t,]')
    cones['File'] = cones['File'].astype(str)

    lane_data = pd.read_csv(lane_data_file, sep='[\t,]')

    for folder, split, pattern in folders:
        #files = getFilesInFolder(folder, '.csv')
        print("-----Video Folder:", folder, "------")
        files = list(filter(lambda f: ('.csv' in f), os.listdir(folder)))
        files = [file for file in files if (not 'cone' in file)]
        if not pattern is None:
            files = [file for file in files if pattern in file]
        print(files)

        for file in files:
            process_file(folder,file, split, lane_data, output_file)

def process_file(folder, file, split, lane_data, output_file):
    """
    Used for batch processing of scripts. May not work properly now.
    """
    try:
        data = pd.read_csv(folder+file, engine='python', delimiter='[\t,]')
        sp = file.split(split)
        if len(sp) == 1:
            sp = os.path.splitext(sp[0])[0].upper()
        else:
            sp = sp[0].upper()
        mvfile = sp +'.MOV'
        scaledata = cones.loc[cones['File']==mvfile]['scale'].dropna().mean()
        framerate = cones.loc[cones['File']==mvfile]['framerate'].dropna().mean()
        print(mvfile, ' - scale:', scaledata)

        dp = DataProcessor(data, pathfile=folder+file, scale=scaledata, framerate=framerate)
        dp.autoslice=False
        
        dp.processData(verbose=verbose)

        try:
            dp.getBounces(chart_folder="../Data/Path Charts/",bounce_baseline=True, verbose=verbose)
        except ValueError as e:
            print("[Error] "+str(e))
            return None
        if len(dp.features.bounces) < 2:
            print("[Error] Too few bounces, skipping")
            return None
        dp.getCrests(chart_folder="../Data/Path Charts/", verbose=verbose)
        dp.fit_to_curves(verbose=verbose)
        dp.get_speeds(verbose=1)
        
        if mvfile in lane_data['Video File'].unique():
            lane_info = lane_data.loc[lane_data['Video File'] == mvfile].iloc[0]
            dp.features.lane = str(lane_info['Lanes normalized'])
            dp.features.angle = str(lane_info['Angle'])
            dp.features.condition = str(lane_info['Condition'])

        dp.write_stats_to_csv(csv_fname=output_file)
        #dp.write_to_mongo()
    except Exception as e:
        #raise e
        print("[Error] in file "+file + " continuing...")
        print(type(e), e)

def process_folders():
    """
    Used for batch processing of scripts. May not work properly now.
    """
    cones = pd.read_csv('../Media/Calhoun Lanes/cone_detections_summary.csv', sep='[\t,]')
    cones['File'] = cones['File'].astype(str)

    folders = [
        ("../Media/Calhoun Trackable/Soccer Launcher Side View/outputs/","_ball", None), #grass
        ("..\\Media\\Calhoun Lanes\\soccer 18 degree kicks\\", "p", None),
        ("..\\Media\\Calhoun Lanes\\soccer 0 degree 55 degree\\", "p", None),
        ("..\\Media\\Calhoun Lanes\\soccer 8 degree\\", "p", None),
        ("..\\Media\\Calhoun Lanes\\v2-Soccer Centerback & Longshot 4-8-22\\outputs_act\\", "id", None),
        ("..\\Media\\Calhoun Lanes\\v2-Soccer Ground and High\\", "-", None)
    ]

    autoprocess_folders(cones, folders, output_file="../Data/auto_results_v10.csv")
    
    exit()

def _process_file(datafile, conefile):
    """
    Used for batch processing of scripts. May not work properly now.
    """
    verbose = False
    testfile = '../Media/Calhoun Lanes/v2-Soccer Centerback & Longshot 4-8-22/outputs_act/IMG_0916id120.csv'
    conefile = '../Media/Calhoun Lanes/cone_detections_summary.csv'

    cones = pd.read_csv('../Media/Calhoun Lanes/cone_detections_summary.csv', sep='[\t,]')
    cones['File'] = cones['File'].astype(str)

    
    dp = DataProcessor(pathfile = testfile)
    dp.processData(verbose=0)
    #dp.noise_frequency(dp.path['Pos X'], dp.path['Time'], verbose=2)
    dp.getBounces(bounce_baseline=True, verbose=3)
    dp.getCrests(chart_folder="../Data/Path Charts/", verbose=0)
    dp.fit_to_curves(verbose=0)
    dp.get_speeds(verbose=0)
    plt.show()
    #autoprocess_folders(cones, folders)

if(__name__ == '__main__'):
    verbose = False
    import argparse
    parser = argparse.ArgumentParser(
        prog="Trackable Data Processor",
        description="Application to ingest the ball tracking data and output stats for the user/app",
    )
    parser.add_argument("input_file", 
                        help="name of csv file to process, requires columns" +
                             "Frame, Pos X, Pos Y, Time, possibly separated" +
                             "object_id in tracking.") 
    parser.add_argument("--output_file", default=None,
                        help="CSV filename to output")
    parser.add_argument("--mongodb", action='store_true')
    parser.add_argument("-v", '--verbose', type=int,
                        help="Level of verbosity, higher is more granular",
                        default=0)
    parser.add_argument("--chartpath", default=None,
                        help="Folder to store chart images")
    parser.add_argument("--bounce_baseline", action='store_false')
    parser.add_argument("--autoslice", action='store_false')
    parser.add_argument("--object_id", type=int,
                        help="ID of object to analyze, zero indexed")
    args = parser.parse_args()

    #testfile = '../Media/Calhoun Lanes/v2-Soccer Centerback & Longshot 4-8-22/outputs_act/IMG_0916id120.csv'
    
    dp = DataProcessor(pathfile = args.input_file, autoslice=args.autoslice)
    
    for id in dp.all_paths['object_id'].unique():
        dp.select_active_object(id)
        dp.processData(verbose=args.verbose, flatten_window=False)
        dp.getBounces(bounce_baseline=True, verbose=args.verbose)
        if (len(dp.features.bounces) > 1):
            dp.getCrests(chart_folder="../Data/Path Charts/", verbose=args.verbose)
            dp.fit_to_curves(verbose=args.verbose)
            dp.get_speeds(verbose=args.verbose)
        
        if args.mongodb:
            dp.write_to_mongo()
        if args.output_file is not None:
            dp.write_stats_to_csv(csv_fname=args.output_file)
    
    if args.verbose > 0:
        plt.show()
    #autoprocess_folders(cones, folders)


        
