#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 11:27:44 2022

@author: jtm545
"""

import pandas as pd
import matplotlib.pyplot as plt

from pyplr.oceanops import OceanOptics


def main():
    try:
    
        oo = OceanOptics.from_first_available()
        wls = oo.wavelengths()
        
        scans_to_average = [1, 5, 10, 50, 100, 500]
        for i, n in enumerate(scans_to_average):
            
            counts, info = oo.measurement(nscans_to_average=n)
            plt.plot(wls, counts+(i*5000), label=n)
        plt.legend(title='Scans averaged')
        plt.title('Scan averaging')
        plt.xlabel('Wavelength (nm)')
        plt.show()
        
        counts = pd.Series(counts)
        windows = [1, 3, 5, 10, 20]
        for i, w in enumerate(windows):
            smoothed = counts.rolling(w, min_periods=1).mean()
            plt.plot(wls, smoothed+(i*5000), lw=1, label=w)
            
        plt.legend(title='Window size')
        plt.title('Boxcar smoothing')
        plt.xlabel('Wavelength (nm)')
        plt.show()


    except KeyboardInterrupt:
        print('> Measurement cancelled by user.')
    
    finally:
        print('> Closing connection to spectrometer.')
        oo.close()

if __name__ == '__main__':
    main()
    
    
# cc = pd.read_table('/Users/jtm545/Projects/BakerWadeBBSRC/hardware/OceanOptics/HL-2000/030410313_CC.LMP',
#               index_col=0,
#               header=None)

# fib = pd.read_table('/Users/jtm545/Projects/BakerWadeBBSRC/hardware/OceanOptics/HL-2000/030410313_FIB.LMP',
#               index_col=0,
#               header=None)