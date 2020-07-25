# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 06:47:41 2019

@author: keigo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def color_R(x):
    if x >= 80:
        red = 255*(np.tanh(1/20*((x-35)*2-255)/2)+1)/2
            
    elif 30 < x < 80:
        red = -100*(np.tanh(1/5*((x+70)*2-255)/2)+1)/2+100
             
    elif x <= 30 :
        red = 100*(np.tanh(1/5*((x+110)*2-255)/2)+1)/2
        
    return red
    
    
def color_G(x):
    if x < 148:
        green = 255*(np.tanh(1/20*((x+30)*2-255)/2)+1)/2
            
    elif 148 <= x < 220:
        green = -255*(np.tanh(1/20*((x-70)*2-255)/2)+1)/2+255
        
    elif x >= 220:
        green = 225*(np.tanh(1/5*((x-107)*2-255)/2)+1)/2+30
        
    return green
    
    
def color_B(x):
    if 200 > x > 45:
        blue = -255*(np.tanh(1/20*((x-5)*2-255)/2)+1)/2+255
            
    elif 45 >= x:
        blue = 255*(np.tanh(1/10*((x+105)*2-255)/2)+1)/2
        
    elif x >= 200:
        blue = 255*(np.tanh(1/5*((x-90)*2-255)/2)+1)/2
        
    return blue

if __name__ == "__main__":
    x = np.arange(0,256,1)
    y_r = []
    y_g = []
    y_b = []
    for i in x:
        y_r = np.append(y_r, color_R(i))
        y_g = np.append(y_g, color_G(i))
        y_b = np.append(y_b, color_B(i))
        
    data = np.vstack([x,y_r,y_g,y_b]).T
    print(data)
    
    plt.plot(x, y_b)
    plt.plot(x, y_r)
    plt.plot(x, y_g)
    
    data = pd.DataFrame(data)
    print(data)

    data.to_csv('rgb_test.csv')