#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:10:12 2019

@author: frog
"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
#import tkinter.messagebox as tkmsg
import numpy as np
#import os
#import glob
import cv2
#import sys
import matplotlib.pyplot as plt
#import math
import pandas as pd
#import csv
#from scipy.linalg import solve 
#import sympy as sym 
import scipy.optimize
#from natsort import natsorted
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import time
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from matplotlib.figure import Figure
import platform

if platform.system() == "Windows":
    try:
        from instrumental.drivers.cameras import uc480
    except:
        pass
    from win32api import GetSystemMetrics
    screenwidth = GetSystemMetrics(0)
    screenheight = GetSystemMetrics(1)
    
#elif platform.system() == Darwin:
    

class CV2:
    def beam_normalize(img_data):   #規格化
        GUI.dark_offset()
        if dark == 21:
            img_data = abs(img_data-dark_data)
        imgmin = np.min(img_data)
        data0 = img_data-imgmin
        imgmax = np.max(data0)
        ndata = data0/imgmax
        
        return ndata
    
    def beam_row_columns(img_data, ndata):  #ピークの行列データ
        global numrow, numcolumn
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(img_data)
        
        numrow = maxLoc[1]
        numcolumn = maxLoc[0]
        datarow = ndata[numrow,:]
        datacolumn = ndata[:,numcolumn]
        dr = pd.Series(datarow)
        dc = pd.Series(datacolumn)
        
        return dr, dc
    
    def tracking(data):
        global numrow,numcolumn
        trackingimg = np.copy(data)
        if varv.get() == True:
            numrow = int(cam_res_h-1-verticalslider.get())
        if varh.get() == True:
            numcolumn = int(horizontalslider.get())
        datarow = data[numrow,:]
        datacolumn = data[:,numcolumn]
        dr = pd.Series(datarow)
        dc = pd.Series(datacolumn)
        #trackingimg[numrow,:] = 0
        #trackingimg[:,numcolumn] = 0
        trackingimg = cv2.line(trackingimg,(0,numrow),(len(datarow)-1,numrow),color=0,thickness=2)
        trackingimg = cv2.line(trackingimg,(numcolumn,0),(numcolumn,len(datacolumn)-1),color=0,thickness=2)
        
        return trackingimg, dr, dc
         
    def beam_peak(data):    #ピーク抽出
        data_peak = data.iloc[data.sub(1).abs().argsort()[:2]]         
        data_index = list(data_peak.index)
        data_peak = (data_index[0]-data_index[1])/2+data_index[1]
        data_peak = round(data_peak)
        
        return data_peak
    
    
    def beam_intensity(dr, dc, dr_peak, dc_peak):   #ピーク中心のビーム強度のデータ
        dr_res = []
        dc_res = []
        dr_peak = np.array(dr_peak, dtype="int64")
        dc_peak = np.array(dc_peak, dtype="int64")
            
        for i in range(-dr_peak+2,cam_res_w-dr_peak+2):
            dr_res.append(i)
                
        for j in range(-dc_peak+2,cam_res_h-dc_peak+2):
            dc_res.append(j)
            
        dr_data = pd.DataFrame({"row_num" : dr_res, "intensity" : dr})
        dr_data = dr_data.set_index("row_num",drop=True)
            
        dc_data = pd.DataFrame({"col_num" : dc_res, "intensity" : dc})
        dc_data = dc_data.set_index("col_num",drop=True)
        
        return dr_data, dc_data
    
    def beam_size(linedata, linedata_peak, percent_of_intensity):  #ビームサイズ算出(px)
        linedata_cut_0 = linedata[0:linedata_peak-1]
        linedata_cut_1 = linedata[linedata_peak-1:len(linedata)]
        
        linedata_size_0 = linedata_cut_0.iloc[linedata_cut_0.sub(percent_of_intensity).abs().argsort()[:1]]
        linedata_size_1 = linedata_cut_1.iloc[linedata_cut_1.sub(percent_of_intensity).abs().argsort()[:1]]
        
        linedata_size_index_0 = list(linedata_size_0.index)
        linedata_size_index_1 = list(linedata_size_1.index)
        
        linedata_size = linedata_size_index_1[0] - linedata_size_index_0[0]
        
        return linedata_size
    
    def from_pixel_to_beam_width(pixel_width, unit):
        if "C1284R13C" in cam_name:
            if "mm" in unit:
                beam_width = 3.6*10**(-3)*pixel_width
            elif "um" in unit:
                beam_width = 3.6*pixel_width
        elif "C1285R12M" in cam_name:
            if "mm" in unit:
                beam_width = 5.2*10**(-3)*pixel_width
            elif "um" in unit:
                beam_width = 5.2*pixel_width
        elif "test" in cam_name:
            if "mm" in unit:
                beam_width = 1.4*10**(-3)*pixel_width
            elif "um" in unit:
                beam_width = 1.4*pixel_width
        else:
            beam_width = pixel_width
        beam_width = round(beam_width, 3)
        
        return beam_width
    
    def pixel_to_realsize(pixel_data, unit):
        realsize = []
        for i in pixel_data.index.values:
            realsize_data = CV2.from_pixel_to_beam_width(i, unit)
            realsize.append(realsize_data)
    
        return realsize
    
    def fitting(data):
        n = len(data)
        
        a11 = n
        a12 = data.index.values.sum()
        a13 = data.index.values.sum()**2
        
        a21 = data.index.values.sum()
        a22 = data.index.values.sum()**2
        a23 = data.index.values.sum()**3
        
        a31 = data.index.values.sum()**2
        a32 = data.index.values.sum()**3
        a33 = data.index.values.sum()**4
        
        b11 = np.sum(np.log(data.values+1))
        b12 = np.sum(data.index.values*np.log(data.values+1).T)
        b13 = np.sum(data.index.values**2*np.log(data.values+1).T)
        
        #A = np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        #B = np.array([b11,b12,b13])
        
        #X= solve(A,B)
        
        #Ainv = np.linalg.inv(A)
        
        #X = np.dot(Ainv,B)
        
        #X = np.linalg.solve(A,B)
        
        
        a,b,c = sym.symbols("a b c")
        eqn1 = a11*a + a12*b + a13*c - b11
        eqn2 = a21*a + a22*b + a23*c - b12
        eqn3 = a31*a + a32*b + a33*c - b13
        
        #X = solve()
        
        X = sym.solve([eqn1,eqn2,eqn3])
        #print(X)
        
        return X
    
    def sigmoid(x, gain=1, offset_x=0):
        return (255*(np.tanh(((1/100*x+offset_x)*gain)/2)+1)/2)
        
    
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
    
    
    def beam_color(img):
        #color_data = [color_RGB(x*1/256) for x in range(0,256)]
        #color_data = np.array(color_data)
        #color_data = color_data*255
        #color_data = np.array(color_data, dtype="uint8")
        #color_data_list = list(color_data)
        #color_data = list(color_data)
        #look_up_table_color = np.ones((256, 1), dtype = 'uint8' ) * 0
        #for i in range(256):
        #img_r = np.empty((1,256), np.uint8)
        lut_r = np.ones((256, 1), dtype = 'uint8' ) * 0
        lut_g = np.ones((256, 1), dtype = 'uint8' ) * 0
        lut_b = np.ones((256, 1), dtype = 'uint8' ) * 0
        for i in range(256):
            lut_r[i][0] = CV2.color_R(i)
            lut_g[i][0] = CV2.color_G(i)
            lut_b[i][0] = CV2.color_B(i)
        img_r = cv2.LUT(img, lut_r)
        img_g = cv2.LUT(img, lut_g)
        img_b = cv2.LUT(img, lut_b)
        img_bgr = cv2.merge((img_b, img_g, img_r))
        img_rgb = cv2.merge((img_r, img_g, img_b))
        
        return img_bgr, img_rgb
        
        
    def pause_plot(xdata, ydata):
        fig, ax = plt.subplots(1, 1)
        x = xdata
        y = ydata
        lines = ax.plot(x, y)
        
        return lines
        
    def knife_edge(img, axis, from_, to_, unit): #axis Xが0, Yが1
        img_sum = np.sum(img)
        img_line_sum = np.sum(img, axis=axis)
        line_sum = []
        line_value = []
        for i in np.arange(0, len(img_line_sum), 1):
            line_value = np.append(line_value, img_line_sum[i])
            line_sum = np.append(line_sum, np.sum(line_value))
            
        knifeedge_data = line_sum/img_sum
        knifeedge_data = pd.Series(knifeedge_data)
        
        knifeedge_size_0 = knifeedge_data.iloc[knifeedge_data.sub(from_).abs().argsort()[:1]]
        knifeedge_size_1 = knifeedge_data.iloc[knifeedge_data.sub(to_).abs().argsort()[:1]]
        #knifeedge_size_index = list(knifeedge_size_0.index)
        knifeedge_size = np.array(knifeedge_size_1.index) - np.array(knifeedge_size_0.index)
        knifeedge_size_actual = CV2.from_pixel_to_beam_width(int(knifeedge_size), unit)
        
        return knifeedge_size_actual
        #line = []
        #line_list = []
        #for i in np.arange(0, len(line_sum)-1, 1):
            #line = line_sum[i+1] - line_sum[i]
            #line_list = np.append(line_list, line)
    
    def beam_intensity_img(img, linedata, axis): #axis Xは0 Yは1
        linedata = linedata * 255
        linedata = np.array(linedata, dtype="uint8")
        drawline_data = []
        for i in np.arange(0, len(linedata), 1):
            if axis == 0:
                drawline_data = np.append(drawline_data,[i,img.shape[axis]-linedata[i]])
            elif axis == 1:
                drawline_data = np.append(drawline_data,[linedata[i],i])
        drawline_data = np.array(drawline_data, dtype="int")
        drawline_data = drawline_data.reshape(-1,1,2)
        img = cv2.polylines(img, [drawline_data], False, color=255, thickness=2)
        img = np.array(img, dtype="uint8")
        
        return img

class GUI:
    def setup():
        global imgcanvas,imgcap
        global root,subframe11,subframe21
        global fnamebox
        global verticalslider,horizontalslider
        global varv,varh
        global autocorrelator_
        global width, height, resize
        global mainframe21,mainframe31
        global mainframe, mainframe11
        
        root = tk.Tk()
        root.title("BeamProfiler")
        #resolution = "%sx%s" % (screenwidth, screenheight)
        resolution = "1280x720"
        if resolution == "1920x1080":
            root.geometry("1920x1080")
            width = 640
            #height = 512
            height = 480
        elif resolution in ("1366x768", "1600x900", "1280x720"):
            root.geometry("1000x600")
            width = int(640/4*3)
            #height = int(512/4*3)
            height = int(480/4*3)
            
        root.minsize(540, 100)
        
        mainframe = ttk.Frame(root, height=800, width=800)
        mainframe.grid(row=1, column=1, rowspan=2, sticky="nw", pady=10)
        
        mainframe11 = ttk.Frame(mainframe, height=20, width=300)
        mainframe11.grid(row=1, column=1, columnspan=2, sticky="nswe")
        
        mainframe_ = ttk.Frame(mainframe, height=400, width=800)
        mainframe_.grid(row=2, column=1, rowspan=2, columnspan=2, sticky="nw", pady=10)
        
        mainframe21 = ttk.Frame(mainframe_, height=400, width=10)
        mainframe21.grid(row=2, column=1, sticky="nw")
        
        verticalslider = ttk.Scale(mainframe21, from_=cam_res_h-1, to=0, length=height+5, orient="v")
        verticalslider.pack()
        
        mainframe22 = ttk.Frame(mainframe_, height=400, width=550)
        mainframe22.grid(row=2, column=2, sticky="nw")
        
        imgcanvas = tk.Canvas(mainframe22, width=width, height=height)
        imgcanvas.grid(row=1, column=1, sticky="nwe")
        imgcap = tk.Label(imgcanvas)
        imgcap.grid(row=1, column=1, sticky="nwe")
        
        #mainframe23 = ttk.Frame(mainframe, height=400, width=50)
        #mainframe23.grid(row=2, column=3, sticky="nw")
        
        mainframe31 = ttk.Frame(mainframe_, height=50, width=550)
        mainframe31.grid(row=3, column=2, sticky="nw")
        
        horizontalslider = ttk.Scale(mainframe31, from_=0, to=cam_res_w-1, length=width+30, orient="h")
        horizontalslider.pack()
        
        mainframe41 = ttk.Frame(mainframe, height=50, width=600)
        mainframe41.grid(row=4, column=1, columnspan=2, sticky="sw")
        
        style = ttk.Style()
        style.configure("style.TButton", font=("",10,"bold"))
        
        folderbutton = ttk.Button(mainframe41, text="Save as", command=GUI_menu.savefile, style="style.TButton")
        folderbutton.grid(row=1, column=3, padx=50)
    
        fnamebox = ttk.Entry(mainframe41, width=40)
        fnamebox.grid(row=1, column=1, columnspan=2, padx=20)
        
        #subframe = ttk.Frame(root, height=600, width=500)
        #subframe.grid(row=1, column=2, sticky="nswe")
        
        subframe11 = ttk.Notebook(root, width=500, height=150)
        subframe11.grid(row=1, column=2, sticky="nse")
        
        GUI.beamwidth_frame(subframe11)
        GUI.pulse_duration_frame(subframe11)
        
        #style.configure("style.TCheckbutton", font=("",10,"bold"))
        
        subframe21 = ttk.Notebook(root, width=500, height=400)
        subframe21.grid(row=2, column=2, sticky="nw")
        
        GUI.intensity_graph_frame(subframe21)
        GUI.pulse_duration_graph_frame(subframe21)
        
        root.grid_columnconfigure(0, weight = 1)
        root.grid_columnconfigure(1, weight = 1)
        
        mainframe.grid_columnconfigure(0, weight = 1)
        mainframe.grid_columnconfigure(1, weight = 1)
        
        resize=1
        
        root.bind("<Configure>", GUI.resize_window)
        
    def resize_window(slider):
        global width,height,resize
        #global verticalslider,horizontalslider,mainframe21,mainframe31
        print(root.winfo_width())
        if root.winfo_width() <= 890:
            #subframe.grid_forget()
            width = int(640/4*3)
            height = int(480/4*3)
            if resize != 0:                
                GUI.change_scale()
                GUI.change_layout()
                resize=0
        
        elif 890 < root.winfo_width() <= 960:
            #subframe.grid_forget()
            width = int(640/4*3*0.7)
            height = int(480/4*3*0.7)
            if resize != 1:                    
                GUI.change_scale()
                if resize == 0:
                    GUI.change_layout_()
                resize=1
                
        elif 960 < root.winfo_width() <= 990:
            #subframe.grid_forget()
            width = int(640/4*3*0.8)
            height = int(480/4*3*0.8)
            if resize != 2:                    
                GUI.change_scale()
                if resize == 0:
                    GUI.change_layout_()
                resize=2
        
        elif 990 < root.winfo_width() <= 1040:
            #subframe.grid_forget()
            width = int(640/4*3*0.9)
            height = int(480/4*3*0.9)
            if resize != 3:                    
                GUI.change_scale()
                if resize == 0:
                    GUI.change_layout_()
                resize=3
                            
        elif root.winfo_width() > 1040:
            #subframe.grid(row=1, column=2, sticky="nswe")
            width = int(640/4*3)
            height = int(480/4*3)
            if resize != 4:                    
                GUI.change_scale()
                if resize == 0:
                    GUI.change_layout_()
                resize=4
        
    def change_scale():
        global verticalslider,horizontalslider,mainframe21,mainframe31
        verticalslider.pack_forget()
        verticalslider = ttk.Scale(mainframe21, from_=cam_res_h-1, to=0, length=height+5, orient="v")
        verticalslider.pack()
        horizontalslider.pack_forget()
        horizontalslider = ttk.Scale(mainframe31, from_=0, to=cam_res_w-1, length=width+30, orient="h")    
        horizontalslider.pack()
        
    def change_layout():
        global mainframe,subframe,mainframe51
        mainframe.grid_forget()
        mainframe.grid(row=1, column=1,sticky="nswe")
        subframe11.grid_forget()
        subframe21.grid_forget()
        #mainframe51 = ttk.Notebook(mainframe, width=400, height=150)
        #mainframe51.grid(row=5, column=1, columnspan=2, sticky="nswe")
        subframe11.grid(row=2, column=1, sticky="ns")
        #GUI.beamwidth_frame(mainframe51)
        
    def change_layout_():
        global mainframe,subframe
        mainframe.grid(row=1, column=1, rowspan=2, sticky="nswe")
        subframe11.grid(row=1, column=2, sticky="nswe")
        subframe21.grid(row=2, column=2, sticky="nswe")
        #mainframe51.grid_forget()
        #GUI.beamwidth_frame(mainframe51)
        
    def button(frame01):
        trackingphoto = tk.PhotoImage(file="tracking.png")
        trackingphoto = trackingphoto.subsample(5)
        trackingbutton = ttk.Button(frame01, image=trackingphoto, command=GUI.tracking_button, style="style.TButton")
        trackingbutton.image = trackingphoto
        trackingbutton.grid(row=1, column=1)
        
        darkphoto = tk.PhotoImage(file="offset.png")
        darkphoto = darkphoto.subsample(5)
        darkbutton = ttk.Button(frame01, image=darkphoto, command=GUI.dark, style="style.TButton")
        darkbutton.image = darkphoto
        darkbutton.grid(row=1, column=2)
        
        axisphoto = tk.PhotoImage(file="axis_px.png")
        axisphoto = axisphoto.subsample(5)
        axisbutton = ttk.Button(frame01, image=axisphoto, command=GUI_menu.switch_state, style="style.TButton")
        axisbutton.image = axisphoto
        axisbutton.grid(row=1, column=3)
        
    def beamwidth_frame(subframe11):
        global Static2,Static3,Static4
        global Static21,Static22,Static31,Static32,Static41,Static42 
        tab11 = tk.Canvas(subframe11)
        subframe11.add(tab11, text="Beam state", padding=10)
        
        subframe111 = ttk.Frame(tab11, height=60, width=100)
        subframe111.grid(row=1, column=1, sticky="nw", pady=20)
        
        Static1 = ttk.Label(subframe111, text='BeamWidth', font=("",10,"bold"))
        Static1.grid(row=1, column=1 ,sticky="w", pady=0)
        Static11 = ttk.Label(subframe111, text='X', font=("",10,"bold"))
        Static11.grid(row=1, column=2, padx=20)
        Static12 = ttk.Label(subframe111, text='Y', font=("",10,"bold"))
        Static12.grid(row=1, column=3 ,padx=20)
        Static2 = ttk.Label(subframe111, text='13.5% of peak (px)', font=("",10,"bold"))
        Static2.grid(row=2, column=1,sticky="w")
        Static3 = ttk.Label(subframe111, text='50.0% of peak (px)', font=("",10,"bold"))
        Static3.grid(row=3, column=1,sticky="w")
        Static4 = ttk.Label(subframe111, text='Peak position (px)', font=("",10,"bold"))
        Static4.grid(row=4, column=1,sticky="w")
        #Static4 = ttk.Label(frame11, text='knife edge 10/90 (mm)', font=("",10,"bold"))
        #Static4.grid(row=4, column=1)
        #Static5 = ttk.Label(frame11, text='knife edge 20/80 (mm)', font=("",10,"bold"))
        #Static5.grid(row=5, column=1)
            
        X_size_e2, Y_size_e2 = 0,0
        X_size_FWHM, Y_size_FWHM = 0,0
        X_peak_position, Y_peak_position = 0,0
        #X_knife_edge_10_90, Y_knife_edge_10_90 = 0,0
        #X_knife_edge_20_80, Y_knife_edge_20_80 = 0,0
        
        Static21 = ttk.Label(subframe111, text=X_size_e2, font=("",10,"bold"))
        Static21.grid(row=2, column=2)
        Static22 = ttk.Label(subframe111, text=Y_size_e2, font=("",10,"bold"))
        Static22.grid(row=2, column=3)
        Static31 = ttk.Label(subframe111, text=X_size_FWHM, font=("",10,"bold"))
        Static31.grid(row=3, column=2)
        Static32 = ttk.Label(subframe111, text=Y_size_FWHM, font=("",10,"bold"))
        Static32.grid(row=3, column=3)
        Static41 = ttk.Label(subframe111, text=X_peak_position, font=("",10,"bold"))
        Static41.grid(row=4, column=2)
        Static42 = ttk.Label(subframe111, text=Y_peak_position, font=("",10,"bold"))
        Static42.grid(row=4, column=3)
        #Static41 = ttk.Label(frame11, text=X_knife_edge_10_90, font=("",10,"bold"))
        #Static41.grid(row=4, column=2)
        #Static42 = ttk.Label(frame11, text=Y_knife_edge_10_90, font=("",10,"bold"))
        #Static42.grid(row=4, column=3)
        #Static51 = ttk.Label(frame11, text=X_knife_edge_20_80, font=("",10,"bold"))
        #Static51.grid(row=5, column=2)
        #Static52 = ttk.Label(frame11, text=Y_knife_edge_20_80, font=("",10,"bold"))
        #Static52.grid(row=5, column=3)
        
    def intensity_graph_frame(subframe21):
        global fig1,fig2
        global ax1,ax2
        global canvas1,canvas2
        tab21 = tk.Canvas(subframe21)
        subframe21.add(tab21, text="Intensity graph")
        
        subframe211 = ttk.Frame(tab21, height=200, width=500)
        subframe211.grid(row=1, column=1, sticky="nswe", padx=10, pady=10)
        
        fig1 = Figure(figsize=(8, 3), dpi=60)
        fig1.subplots_adjust(bottom=0.2)
        ax1 = fig1.add_subplot(111)
        #fig1.tight_layout()
        ax1.set_xlabel("X (px)")
        ax1.set_ylabel("Intensity (arb.units)")
        
        canvas1 = FigureCanvasTkAgg(fig1, master=subframe211)
        canvas1.get_tk_widget().grid(column=1, row=1, sticky="nswe")
        canvas1._tkcanvas.grid(column=1, row=1, sticky="nswe")
        
        subframe212 = ttk.Frame(tab21, height=200, width=500)
        subframe212.grid(row=2, column=1, sticky="nswe", padx=10)
        
        fig2 = Figure(figsize=(8, 3), dpi=60)
        fig2.subplots_adjust(bottom=0.2)
        #fig2.patch.set_alpha(0)
        ax2 = fig2.add_subplot(111)
        #fig2.tight_layout()
        ax2.set_xlabel("Y (px)",labelpad=None)
        ax2.set_ylabel("Intensity (arb.units)",labelpad=None)
        
        canvas2 = FigureCanvasTkAgg(fig2, master=subframe212)
        canvas2.get_tk_widget().grid(column=1, row=1, sticky="nswe")
        canvas2._tkcanvas.grid(column=1, row=1, sticky="nswe")
        
    def pulse_duration_frame(subframe11):
        global Static_a11
        tab12 = tk.Canvas(subframe11)
        subframe11.add(tab12, text="Autocorrelator")
        
        subframe121 = ttk.Frame(tab12, height=60, width=100)
        subframe121.grid(row=1, column=1, sticky="nswe")
        
        FWHM_t = 0
                
        Static_a11 = ttk.Label(subframe121, text="%s fs" % FWHM_t, font=("",80,"bold"))
        Static_a11.grid(row=1, column=1, sticky="nw", padx=10)
        
    def pulse_duration_graph_frame(subframe21):
        global subframe222,subframe223
        global dxbox
        tab22 = tk.Canvas(subframe21)
        subframe21.add(tab22, text="Autocorrelator")
        
        subframe221 = ttk.Frame(tab22, height=50, width=400)
        subframe221.grid(row=1, column=1, sticky="nswe", pady=10)
        
        startbutton = ttk.Button(subframe221, text="Base", command=GUI.calculate, style="style.TButton")
        startbutton.grid(row=1, column=1, padx=10, sticky="nw")
        
        dxbox = ttk.Entry(subframe221, width=10)
        dxbox.grid(row=1, column=2, sticky="n")
        
        dxlabel = ttk.Label(subframe221, text="mm", font=("",10,"bold"))
        dxlabel.grid(row=1, column=3, sticky="n")
        
        dxbutton = ttk.Button(subframe221, text="Second", command=GUI.autocorrelator, style="style.TButton")
        dxbutton.grid(row=1, column=4, padx=10, sticky="n")
        
        acsavebutton = ttk.Button(subframe221, text="Save", command=GUI_menu.acsavefile, style="style.TButton")
        acsavebutton.grid(row=1, column=5, padx=10, sticky="n")
        
        subframe222 = ttk.Frame(tab22, height=60, width=300)
        subframe222.grid(row=2, column=1, sticky="nw")
        
        subframe223 = ttk.Frame(tab22, height=60, width=300)
        subframe223.grid(row=3, column=1, sticky="nw")
        
    def cam_setup():
        global cap,cam,cam_name
        global frame
        global exposuretime
        try:
            cam = uc480.UC480_Camera()
        except:
            pass
        cam_name = cam.model
        cam_name = str(cam_name)
        cam.start_capture()
        #frame = cam.grab_image(timeout='None', copy=True,width=640,height=480)
        exposuretime = "0.2ms"
        frame = cam.start_live_video(framerate=None, exposure_time=exposuretime)
        #frame = cam.get_captured_image(timeout='10s', copy=True)
        #camera_id = 1
        #cap = cv2.VideoCapture(camera_id)
        #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        #cap.set(cv2.CAP_PROP_EXPOSURE, shutterspeed)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1)
    
    
    def cam_setup_test(camera_id):
        global cam, cam_name, cam_res
        global cam_res_w, cam_res_h
        global exposuretime
        cam = cv2.VideoCapture(camera_id)
        cam_res = "1024x768"
        cam_res = cam_res.split("x")
        cam_res_w = int(cam_res[0])
        cam_res_h = int(cam_res[1])
        #cam_res_w = 1024
        #cam_res_h = 768
        #cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
        exposuretime = -5
        cam.set(cv2.CAP_PROP_EXPOSURE, exposuretime)
        #print(cam.get(cv2.CAP_PROP_EXPOSURE))
        #cam_res_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        #cam_res_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #print(cam_res_w, cam_res_h)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_res_w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res_h)
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        cam_name = "test"
        time.sleep(1)
        
    def cam_select():
        cam_list = [] 
        for i in np.arange(0, 6, 1):
            cam = cv2.VideoCapture(i)
            if cam.isOpened() == True:
                cam_list.append(i)
            elif cam.isOpened() == False:
                pass
            
        return cam_list
          
    def beamprofiler_img():
        global frame,img,img_norm
        global X,Y
        global beamimg,beamimg_save,barimg_save,save_img
        global dark,trackingon
        #ret, frame = cam.read()
        #frame = cam.get_captured_image(timeout='10s', copy=True)
        #frame = cam.latest_frame(copy=True)
        #frame = cv2.flip(frame, 1)
        if "C1284R13C" in cam_name:
            frame = cam.latest_frame(copy=True)
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif "C1285R12M" in cam_name:
            frame = cam.latest_frame(copy=True)
            frame = cv2.flip(frame, 1)
            img = frame
        elif "test" in cam_name:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_norm = CV2.beam_normalize(img)
        X, Y = CV2.beam_row_columns(img, img_norm)
        if trackingon == 1:
            trackingimg, X, Y = CV2.tracking(img_norm)
            img_norm = trackingimg
        img_beamintensity = img_norm * 255
        img_beamintensity = np.array(img_beamintensity, dtype="uint8")
        img_beamintensity = CV2.beam_intensity_img(img_beamintensity, X, 0)
        img_beamintensity = CV2.beam_intensity_img(img_beamintensity, Y, 1)
        #img_beamintensity = cv2.putText(img_beamintensity, "100", (10,500), color=0, fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        beamimg_save,beamimg = CV2.beam_color(img_beamintensity)
        #beamimg_save,_ = CV2.beam_color(save_img)
        beam_img = cv2.resize(beamimg,(width,height))
        barimg_save,bar_img = GUI.colorbar()
        beam_img = cv2.hconcat([beam_img,bar_img])
        beam_img = Image.fromarray(beam_img)
        beam_img_tk = ImageTk.PhotoImage(image=beam_img, master=imgcanvas)
        imgcap.beam_img_tk = beam_img_tk
        beam_img = imgcap.configure(image=beam_img_tk)
        
        imgcap.after(100, GUI.beamprofiler_img)
    
    def plotter():
        global fig1,fig2
        global ax1,ax2
        global canvas1,canvas2
        global X,Y
        global realsize_X, realsize_Y
        
        try:
            ax1.cla()
            ax2.cla()
        except:
            pass
        state = var.get()
        
        if state == 0:
            ax1.plot(X.index.values, X.values)
            ax1.set_xlabel("X (px)")
            ax1.set_xlim(0,cam_res_w)
            ax1.set_xticks(np.arange(0,cam_res_w+1,100))
        elif state == 1:
            realsize_X = CV2.pixel_to_realsize(X, "mm")
            ax1.plot(realsize_X, X.values)
            ax1.set_xlabel("X (mm)")
            ax1.set_xlim(0,realsize_X[cam_res_w-1])
            #ax1.set_xticks(np.arange(0,realsize_X[cam_res_h-1]+1,100))
        elif state == 2:
            realsize_X = CV2.pixel_to_realsize(X, "um")
            ax1.plot(realsize_X, X.values)
            ax1.set_xlabel("X (um)")
            ax1.set_xlim(0,realsize_X[cam_res_w-1])
        ax1.set_ylabel("Intensity (arb.units)")
        ax1.set_ylim(0,1.2)
        ax1.set_yticks(np.arange(0,1.2+0.2,0.2))
        
        YY = Y.iloc[::-1]
        if state == 0:
            #ax2.plot(YY.values, Y.index.values)
            ax2.plot(Y.index.values, YY.values)
            ax2.set_xlabel("Y (px)",labelpad=None)
            ax2.set_xlim(0,cam_res_h)
            ax2.set_xticks(np.arange(0,cam_res_h+1,100))
        elif state == 1:
            realsize_Y = CV2.pixel_to_realsize(YY, "mm")
            realsize_Y.reverse()
            #ax2.plot(YY.values, realsize_Y)
            ax2.plot(realsize_Y, YY.values)
            ax2.set_xlabel("Y (mm)",labelpad=None)
            ax2.set_xlim(0,realsize_Y[cam_res_h-1])
            #ax2.set_yticks(np.arange(0,realsize_Y[cam_res_h-1]+1,100))
        elif state == 2:
            realsize_Y = CV2.pixel_to_realsize(YY, "um")
            realsize_Y.reverse()
            #ax2.plot(YY.values, realsize_Y)
            ax2.plot(realsize_Y, YY.values)
            ax2.set_xlabel("Y (um)",labelpad=None)
            ax2.set_xlim(0,realsize_Y[cam_res_h-1])
        ax2.set_ylabel("Intensity (arb.units)",labelpad=None)
        ax2.set_ylim(0,1.2)
        ax2.set_yticks(np.arange(0,1.2+0.2,0.2))
        
        canvas1.draw()
        canvas2.draw()
        
        root.after(100, GUI.plotter)
        
    def beam_width():
        global X_peak_position, Y_peak_position
        global X_size_e2,Y_size_e2
        global X_size_FWHM,Y_size_FWHM
        global X_knife_edge_10_90,Y_knife_edge_10_90
        global X_knife_edge_20_80,Y_knife_edge_20_80
        global knifeedge_count
        X_peak = CV2.beam_peak(X)
        Y_peak = CV2.beam_peak(Y)
        try:
            X_size_e2_px = CV2.beam_size(X, X_peak, 1/np.exp(2))
            Y_size_e2_px = CV2.beam_size(Y, Y_peak, 1/np.exp(2))
            X_size_FWHM_px = CV2.beam_size(X, X_peak, 0.5)
            Y_size_FWHM_px = CV2.beam_size(Y, Y_peak, 0.5)
        except:
            pass
        
        state = var.get()
        if state == 0:
            Static2.configure(text='13.5% of peak (px)', font=("",10,"bold"))
            Static3.configure(text='50.0% of peak (px)', font=("",10,"bold"))
            Static4.configure(text='Peak position (px)', font=("",10,"bold"))
            X_size_e2 = X_size_e2_px
            Y_size_e2 = Y_size_e2_px
            X_size_FWHM = X_size_FWHM_px
            Y_size_FWHM = Y_size_FWHM_px
            X_peak_position = X_peak
            Y_peak_position = Y_peak
        elif state == 1:
            Static2.configure(text='13.5% of peak (mm)', font=("",10,"bold"))
            Static3.configure(text='50.0% of peak (mm)', font=("",10,"bold"))
            Static4.configure(text='Peak position (mm)', font=("",10,"bold"))
            X_size_e2 = CV2.from_pixel_to_beam_width(X_size_e2_px, "mm")
            Y_size_e2 = CV2.from_pixel_to_beam_width(Y_size_e2_px, "mm")
            X_size_FWHM = CV2.from_pixel_to_beam_width(X_size_FWHM_px, "mm")
            Y_size_FWHM = CV2.from_pixel_to_beam_width(Y_size_FWHM_px, "mm")
            X_peak_position = CV2.from_pixel_to_beam_width(X_peak, "mm")
            Y_peak_position = CV2.from_pixel_to_beam_width(Y_peak, "mm")
        elif state == 2:
            Static2.configure(text='13.5% of peak (um)', font=("",10,"bold"))
            Static3.configure(text='50.0% of peak (um)', font=("",10,"bold"))
            Static4.configure(text='Peak position (um)', font=("",10,"bold"))
            X_size_e2 = CV2.from_pixel_to_beam_width(X_size_e2_px, "um")
            Y_size_e2 = CV2.from_pixel_to_beam_width(Y_size_e2_px, "um")
            X_size_FWHM = CV2.from_pixel_to_beam_width(X_size_FWHM_px, "um")
            Y_size_FWHM = CV2.from_pixel_to_beam_width(Y_size_FWHM_px, "um")
            X_peak_position = CV2.from_pixel_to_beam_width(X_peak, "um")
            Y_peak_position = CV2.from_pixel_to_beam_width(Y_peak, "um")
        else:
            pass
        
        Static21.configure(text=X_size_e2, font=("",10,"bold"))
        Static22.configure(text=Y_size_e2, font=("",10,"bold"))
        Static31.configure(text=X_size_FWHM, font=("",10,"bold"))
        Static32.configure(text=Y_size_FWHM, font=("",10,"bold"))
        Static41.configure(text=X_peak_position, font=("",10,"bold"))
        Static42.configure(text=Y_peak_position, font=("",10,"bold"))
        
        #knifeedge_count = knifeedge_count + 1
        
        #if knifeedge_count == 2:
            #X_knife_edge_10_90 = CV2.knife_edge(img, 0, 0.1, 0.9)
            #Y_knife_edge_10_90 = CV2.knife_edge(img, 1, 0.1, 0.9)
            #X_knife_edge_20_80 = CV2.knife_edge(img, 0, 0.2, 0.8)
            #Y_knife_edge_20_80 = CV2.knife_edge(img, 1, 0.2, 0.8)
            #Static41.configure(text=X_knife_edge_10_90, font=("",10,"bold"))
            #Static42.configure(text=Y_knife_edge_10_90, font=("",10,"bold"))
            #Static51.configure(text=X_knife_edge_20_80, font=("",10,"bold"))
            #Static52.configure(text=Y_knife_edge_20_80, font=("",10,"bold"))
            #knifeedge_count = 0
            
        
        
        #if autocorrelator_ == 1:
            #pix2 = X_peak
            #Static_a21.configure(text=pix2)
        
        root.after(100, GUI.beam_width)
        
    def colorbar():
        num = np.linspace(255,0,256,dtype="uint8")
        num = np.tile(num,(25,1))
        barimg_bgr,barimg_rbg = CV2.beam_color(num.T)
        #bar_img = cv2.resize(bar_img, dsize=None, fx=1, fy=1.5)
        barimg_bgr = cv2.resize(barimg_bgr, dsize=(40, cam_res_h))
        barimg_rbg = cv2.resize(barimg_rbg, dsize=(25, height))
        #bar_img = Image.fromarray(bar_img)
        #bar_img_tk = ImageTk.PhotoImage(image=bar_img, master=barcanvas)
        #barimg.bar_img_tk = bar_img_tk
        #barimg = barimg.configure(image=bar_img_tk)
        
        return barimg_bgr, barimg_rbg
        
    def dark():
        global dark,dark_data
        dark = 1
        #dark_data = img
        
    def dark_offset():
        global dark_data, dark
        if dark == 1:
            #tkmsg.showinfo("Info", "Please wait a moment")
            GUI.waitdialog("Wait a moment")
            dark_data = img
            dark = dark + 1
        elif 2 <= dark < 20:
            dark_data = np.dstack([dark_data, img])
            dark = dark + 1
        elif dark == 20:
            dark_data = np.dstack([dark_data, img])
            dark_data = dark_data.mean(axis=2)
            dark = dark + 1
            try:
                msgdialog.destroy()
            except:
                pass
        else:
            pass
        
    def waitdialog(message):
        global msgdialog
        msgdialog = tk.Toplevel(root)
        msgdialog.transient()
        msgdialog.title('Info')
        tk.Label(msgdialog, text=message, font=("",20,"bold")).grid(padx=20, pady=20)
        
        return msgdialog
        
    def exposure_time():
        global exposuretime
        exposuretime = exposuretimebox.get()
        if "C1284R13C" in cam_name or "C1285R12M" in cam_name:
            cam.stop_live_video()
            cam.start_live_video(framerate=None, exposure_time="%s ms" % exposuretime)
            time.sleep(0.5)
        elif "test" in cam_name:
            cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cam.set(cv2.CAP_PROP_EXPOSURE,float(exposuretime))
            time.sleep(0.5)
        
    def trigger():
        cam.stop_live_video()
        #cam.set_trigger(mode='hardware', edge='rising')
        cam.blacklevel_offset
        cam.start_live_video(framerate=None)
        time.sleep(0.5)
        
    def tracking_button():
        global trackingon
        if trackingon == 0:
            trackingon = 1
        elif trackingon == 1:
            trackingon = 0
            
    def hsliderbutton():
        if varh.get() == True:
            horizontalslider.set(numcolumn)
            
    def vsliderbutton():
        if varv.get() == True:
            verticalslider.set(cam_res_h-1-numrow)

    def fittingfunc(x,mu,sigma):
            if func == "gaussian":
                return np.exp(-(x-mu)**2 / (2.0*sigma**2))
        
            elif func == "lorentz":
                return sigma**2/(4*(x-mu)**2+sigma**2)
    
    def scipy_fit(xdata,ydata):
        X = np.ravel(xdata)
        Y = np.ravel(ydata)
        try:
            func
        except:
            func = "gaussian"
        
        def fittingfunc(x,mu,sigma):
            if func == "gaussian":
                return np.exp(-(x-mu)**2 / (2.0*sigma**2))
        
            elif func == "lorentz":
                return sigma**2/(4*(x-mu)**2+sigma**2)
    
        params,cov = scipy.optimize.curve_fit(fittingfunc,X,Y)
    
        return params

    def autocorrelator_graph():
        global fig3,fig4
        global ax3,ax4
        fig3 = Figure(figsize=(8, 3), dpi=60)
        fig3.subplots_adjust(bottom=0.4)
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel("X (px)")
        ax3.set_ylabel("Intensity (arb.units)")
        ax3.set_xlim(0,cam_res_w)
        ax3.set_ylim(0,1.2)
        ax3.set_xticks(np.arange(0,cam_res_w+1,100))
        ax3.set_yticks(np.arange(0,1.2+0.2,0.2))
        
        fig4 = Figure(figsize=(8, 3), dpi=60)
        fig4.subplots_adjust(bottom=0.4)
        ax4 = fig4.add_subplot(111)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Intensity (arb.units)")
        #ax4.set_xlim(0,1280)
        #ax4.set_ylim(0,1.2)
        #ax4.set_xticks(np.arange(0,1280+1,100))
        #ax4.set_yticks(np.arange(0,1.2+0.2,0.2))
        
    def calculate():
        global pix1, pix2, pix, X_gaussian
        global acdata1, acdata2
        pix1 = 0
        if pix1 == False: 
            ax3.cla()
        ax3.plot(X.index.values, X.values)
        X_peak = CV2.beam_peak(X)
        Y_peak = CV2.beam_peak(Y)
        if pix1 == False:
            #pix1 = X_peak
            pix = X.index.values
            X_data, Y_data = CV2.beam_intensity(X, Y, X_peak, Y_peak)
            params = GUI.scipy_fit(X_data.index.values, X_data.values)
            X_gaussian = GUI.fittingfunc(X_data.index.values, params[0], params[1])
            X_gaussian = pd.Series(X_gaussian)
            try:
                acdata1
            except:
                acdata1 = []
                acdata2 = beamimg_save
                acdata1 = pd.DataFrame({"Normalized(original)":pd.Series(X), "Fitting(original)":X_gaussian})
            pix1 = CV2.beam_peak(X_gaussian)
            #pix1_peak = X_gaussian.max
            #pix1_size = CV2.beam_size_(X_gaussian, pix1_peak, 1/2**0.5)
            ax3.plot(X.index.values, X_gaussian.values)
            ax3.set_xlabel("X (px)")
            ax3.set_ylabel("Intensity (arb.units)")
            ax3.set_xlim(0,cam_res_w)
            ax3.set_ylim(0,1.2)
            ax3.set_xticks(np.arange(0,cam_res_w+1,200))
            ax3.set_yticks(np.arange(0,1.2+0.2,0.2))
            
            canvas3 = FigureCanvasTkAgg(fig3, master=subframe222)
            canvas3.get_tk_widget().grid(row=1, column=1)
            canvas3._tkcanvas.grid(row=1, column=1)
            
            canvas3.draw()
           
        
            #X_data, Y_data = CV2.beam_intensity(X, Y, X_peak, Y_peak)
            #params = GUI.scipy_fit(X_data.index.values, X_data.values)
            #X_gaussian_2 = GUI.gaussian_fit(X_data.index.values, params[0], params[1])
            #X_gaussian_2 = pd.Series(X_gaussian_2)
            #pix2 = CV2.beam_peak(X_gaussian_2)
            #acdata3 = pd.DataFrame({"Normalized(Second)":pd.Series(X), "Fitting(Second)":X_gaussian_2})
            #acdata1 = pd.concat([acdata1,acdata3], axis=1)
        #pix2_peak = X_gaussian_2.max
        #pix2 = CV2.beam_size_(X_gaussian_2, pix2, 1/2**0.5)
            #time.sleep(0.5)
        
    def autocorrelator():
        global t,FWHM_t
        global autocorrelator_,pix2
        global acdata1, acdata4
        X_peak = CV2.beam_peak(X)
        Y_peak = CV2.beam_peak(Y)
        X_data, Y_data = CV2.beam_intensity(X, Y, X_peak, Y_peak)
        params = GUI.scipy_fit(X_data.index.values, X_data.values)
        X_gaussian_2 = GUI.fittingfunc(X_data.index.values, params[0], params[1])
        X_gaussian_2 = pd.Series(X_gaussian_2)
        pix2 = CV2.beam_peak(X_gaussian_2)
        acdata3 = pd.DataFrame({"Normalized(Second)":pd.Series(X), "Fitting(Second)":X_gaussian_2})
        acdata1 = pd.concat([acdata1,acdata3], axis=1)
        acdata4 = beamimg_save
        dpix = abs(pix1-pix2)
        dx = dxbox.get()
        if func == "gaussian":
            data = 2*float(dx)*10**(-3)/(299792458*dpix)
        elif func == "lorentz":
            data = 1/np.sqrt(2)*float(dx)*10**(-3)/(299792458*dpix)
        t = pix*data
        
        ax4.cla()
        ax4.plot(t, X_gaussian)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Intensity (arb.units)")
        #ax4.set_xlim(0,1280)
        ax4.set_ylim(0,1.2)
        #ax4.set_xticks(np.arange(0,1280+1,200))
        ax4.set_yticks(np.arange(0,1.2+0.2,0.2))
        
        canvas4 = FigureCanvasTkAgg(fig4, master=subframe223)
        canvas4.get_tk_widget().grid(row=1, column=1)
        canvas4._tkcanvas.grid(row=1, column=1)
        
        X_gaussian_ = pd.Series(X_gaussian)
        FWHM = CV2.beam_size(X_gaussian_,pix1,1/2**0.5)
        FWHM_t = FWHM * data * 10**15
        FWHM_t = round(FWHM_t, 1)
        Static_a11.configure(text="%s fs" % FWHM_t, font=("",80,"bold"))
        FWHM_t = pd.DataFrame([FWHM_t], columns=["Pulse duration (fs)"])
        acdata1 = pd.concat([acdata1,FWHM_t], axis=1)
        
        autocorrelator_ = 1
        
    #def tab_create():
        
        

class GUI_menu:
    def mainmenu():
        mainmenu = tk.Menu(root)
        root.config(menu=mainmenu)
        
        filemenu = tk.Menu(mainmenu, tearoff=0)
        mainmenu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Save as", command=GUI_menu.savefile)
        filemenu.add_command(label="Quit", command=GUI_menu.menu_quit)
        #filemenu.add_separator()
        
        toolsmenu = tk.Menu(mainmenu, tearoff=0)
        #settingsmenu = tk.Menu(toolsmenu, tearoff=0)
        cam_select = tk.Menu(toolsmenu, tearoff=0)
        fittingfunction = tk.Menu(toolsmenu, tearoff=0)
        mainmenu.add_cascade(label="Tools", menu=toolsmenu)
        toolsmenu.add_command(label="Settings", command=GUI_menu.settings)
        toolsmenu.add_cascade(label="Camera select", menu=cam_select)
        cam_list = GUI.cam_select()
        for i in cam_list:
            cam_select.add_command(label="%d" % i, command=GUI_menu.switch_cam(i))
        toolsmenu.add_cascade(label="Fitting function", menu=fittingfunction)
        fittingfunction.add_command(label="gaussian", command=GUI_menu.set_gaussian)
        fittingfunction.add_command(label="lorentz", command=GUI_menu.set_lorentz)
        #settingsmenu.add_command(label="Exposure time", command=GUI.exposure_time)
        
        autocorrelatormenu = tk.Menu(mainmenu, tearoff=0)
        mainmenu.add_cascade(label="Autocorrelator", menu=autocorrelatormenu)
    
    def settings():
        global settingsframe       
        settingswindow = tk.Toplevel()
        settingswindow.title("Settings")
        settingswindow.geometry("500x300")
        settingswindow.resizable(0,0)
        #settingswindow.overrideredirect(True)
        settingswindow.grid()
        
        settingsframe = ttk.Notebook(settingswindow, width=500, height=275)
        settingsframe.grid(row=1, column=1, columnspan = 3)
        
        t1,t2,t3,t4,t5 = GUI_menu.createtab()
        
        GUI_menu.tab_camera(t1)
        GUI_menu.tab_graph(t2)
        GUI_menu.tab_capture(t3)
        GUI_menu.tab_savefile(t5)
        
    def createtab():
        t1 = tk.Canvas(settingsframe)
        t2 = tk.Canvas(settingsframe)
        t3 = tk.Canvas(settingsframe)
        t4 = tk.Canvas(settingsframe)
        t5 = tk.Canvas(settingsframe)
        settingsframe.add(t1, text="Camera")
        settingsframe.add(t2, text="Graph")
        settingsframe.add(t3, text="Capture")
        settingsframe.add(t4, text="Autocorrelator")
        settingsframe.add(t5, text="File")
        
        return t1, t2, t3, t4, t5
    
    def tab_camera(t1):
        global exposuretime, exposuretimebox, cam_res_box
        t1frame1 = ttk.Frame(t1, width=500, height=100)
        t1frame1.grid(row=1, column=1, sticky="nw", padx=30, pady=30)
        t1frame2 = ttk.Frame(t1, width=500, height=100)
        t1frame2.grid(row=2, column=1, sticky="nw", padx=30, pady=30)
        
        cam_res_label = ttk.LabelFrame(t1frame1, text="Camera resolution", width=450, height=100)
        cam_res_label.grid(row=1, column=1, sticky="w")
        
        #cam_res_list = ["3264x2448", "2592x1944", "2048x1536", "1600x1200", "1280x960", "1024x768", "800x600", "640x480", "320x240"]
        cam_res_list = ["1600x1200", "1280x960", "1024x768", "800x600", "640x480", "320x240"]
        cam_res_list.reverse()
        cam_res_box = ttk.Combobox(cam_res_label, values=cam_res_list, state="readonly")
        cam_res_box.grid(row=2, column=1, padx=30, pady=10, sticky="w")
        cam_res_box.set("%s" % cam_res)
        
        cam_res_button = ttk.Button(cam_res_label, text="Set", command=GUI_menu.set_cam_res, style="style.TButton")
        cam_res_button.grid(row=2, column=2, padx=10, pady=10, sticky="w")
            
        exposuretimelabel = ttk.LabelFrame(t1frame2, text="Exposuretime", width=450, height=100)
        exposuretimelabel.grid(row=1, column=1, sticky="w")
        
        darkbutton = ttk.Button(exposuretimelabel, text="Offset", command=GUI.dark, style="style.TButton")
        darkbutton.grid(row=2, column=3, padx=10)
            
        if "C1284R13C" in cam_name or "C1285R12M" in cam_name:
            exposuretimebox = ttk.Spinbox(exposuretimelabel, from_=0.1, to=100, increment=0.1)
            exposuretimebox.set("%s" % exposuretime)
        elif "test" in cam_name:
            #exposuretimelist = ["640 ms", "320 ms", "160 ms", "80 ms", "40 ms", "20 ms", "10 ms", "5 ms", "2.5 ms", "1.25 us", "650 um", "312 um", "150 um"]
            #exposuretimebox = ttk.Spinbox(t1frame2, value=exposuretimelist, state="readonly")
            exposuretimebox = ttk.Spinbox(exposuretimelabel, from_=-13, to=-1, increment=1)
            exposuretimebox.set("%s" % exposuretime)
        exposuretimebox.grid(row=2, column=1, pady=10, sticky="w")
            
        exposuretimebutton = ttk.Button(exposuretimelabel, text="Set", command=GUI.exposure_time, style="style.TButton")
        exposuretimebutton.grid(row=2, column=2, padx=10, pady=10, sticky="w")
        
    def tab_graph(t2):
        t2frame1 = ttk.Frame(t2, width=500, height=100)
        t2frame1.grid(row=1, column=1, sticky="nw", padx=30, pady=30)
        
        axislabel = ttk.LabelFrame(t2frame1, text="Axis setting", width=450, height=100)
        axislabel.grid(row=1, column=1, sticky="w")
        
        pixelbutton = ttk.Radiobutton(axislabel, text="Pixel", variable=var, value=0, command=GUI_menu.set_actualsize())
        pixelbutton.grid(row=2, column=1, sticky="w")
        
        actualsizebuttonmm = ttk.Radiobutton(axislabel, text="Actual size (mm)", variable=var, value=1, command=GUI_menu.set_actualsize())
        actualsizebuttonmm.grid(row=3, column=1, sticky="w")
        
        actualsizebuttonum = ttk.Radiobutton(axislabel, text="Actual size (um)", variable=var, value=2, command=GUI_menu.set_actualsize())
        actualsizebuttonum.grid(row=4, column=1, sticky="w")
        
    def tab_capture(t3):
        global varv, varh
        t3frame1 = ttk.Frame(t3, width=500, height=100)
        t3frame1.grid(row=1, column=1, sticky="nw", padx=30, pady=30)
        
        trackingbutton = ttk.Button(t3frame1, text="Tracking", command=GUI.tracking_button, style="style.TButton")
        trackingbutton.grid(row=1, column=2, rowspan=2, padx=20)
        
        style = ttk.Style()
        style.configure("style.TCheckbutton", font=("",10,"bold"))
        
        varv = tk.BooleanVar()
        verticalsliderbutton = ttk.Checkbutton(t3frame1, text="Horizontal slider", variable=varv, command=GUI.vsliderbutton, style="style.TCheckbutton")
        verticalsliderbutton.grid(row=2, column=1, sticky="w")
        
        varh = tk.BooleanVar()
        horizontalsliderbutton = ttk.Checkbutton(t3frame1, text="Vertical slider", variable=varh, command=GUI.hsliderbutton, style="style.TCheckbutton")
        horizontalsliderbutton.grid(row=1, column=1, sticky="w")
        
    def tab_savefile(t5):
        global varraw, varnormal, varimg, vargray, varbar, varintensity
        global var_list_get
        t5frame1 = ttk.Frame(t5, width=500, height=100)
        t5frame1.grid(row=1, column=1, sticky="nw", padx=30, pady=30)
        
        savefilelabel = ttk.LabelFrame(t5frame1, text="Save file", width=450, height=100)
        savefilelabel.grid(row=1, column=1, sticky="w")
        
        try:
            var_list_get
        except:
            varraw = tk.BooleanVar()
            varnormal = tk.BooleanVar()
            varimg = tk.BooleanVar()
            vargray = tk.BooleanVar()
            varbar = tk.BooleanVar()
            varintensity = tk.BooleanVar()
            var_list_get = [True,False,True,False,True,False]
            var_list = [varraw, varnormal, varimg, vargray, varbar, varintensity]
            for i,j in zip(var_list_get, var_list):
                j.set(i)
            
        GUI_menu.get_var()
        
        rawbutton = ttk.Checkbutton(savefilelabel, text="RAW data", command=GUI_menu.get_var, variable=varraw, style="style.TCheckbutton")
        rawbutton.grid(row=2, column=1, sticky="w")
        
        normalbutton = ttk.Checkbutton(savefilelabel, text="Normalized data", command=GUI_menu.get_var, variable=varnormal, style="style.TCheckbutton")
        normalbutton.grid(row=3, column=1, sticky="w")
        
        imgbutton = ttk.Checkbutton(savefilelabel, text="Color image", command=GUI_menu.get_var, variable=varimg, style="style.TCheckbutton")
        imgbutton.grid(row=4, column=1, sticky="w")
        
        graybutton = ttk.Checkbutton(savefilelabel, text="Black-and-white image", command=GUI_menu.get_var, variable=vargray, style="style.TCheckbutton")
        graybutton.grid(row=5, column=1, sticky="w")
        
        barbutton = ttk.Checkbutton(savefilelabel, text="with color bar", command=GUI_menu.get_var, variable=varbar, style="style.TCheckbutton")
        barbutton.grid(row=2, column=2, sticky="w")
        
        intensitybutton = ttk.Checkbutton(savefilelabel, text="with intensity plot", command=GUI_menu.get_var, variable=varintensity, style="style.TCheckbutton")
        intensitybutton.grid(row=3, column=2, sticky="w")
        
    def savefile():
        global fname
        fname = tkfd.asksaveasfile(confirmoverwrite=False, defaultextension=".png", filetypes=[("PNG files",".png"),("JPG files",".jpg"),("BMP files",".bmp"),("TIFF files",".tiff")])
        fnamebox.insert(tk.END,fname.name)
        GUI_menu.get_var()
        filename = fname.name.split(".")
        
        saveimg_norm = CV2.beam_normalize(img)
        X, Y = CV2.beam_row_columns(img, saveimg_norm)
        saveimg = saveimg_norm * 255
        saveimg_gray = np.array(saveimg, dtype="uint8")
        saveimg_color,_ = CV2.beam_color(saveimg_gray)
        saveimg_intensity = CV2.beam_intensity_img(saveimg_gray, X, 0)
        saveimg_intensity = CV2.beam_intensity_img(saveimg_intensity, Y, 1)
        saveimg_intensity,_ = CV2.beam_color(saveimg_intensity)
        
        if varimg.get() == True and varbar.get() == True and varintensity.get() == True:
            beamimg1 = cv2.hconcat([saveimg_intensity,barimg_save])
            cv2.imwrite(fname.name, beamimg1)
        if varimg.get() == True and varbar.get() == True and varintensity.get() == False:
            beamimg2 = cv2.hconcat([saveimg_color,barimg_save])
            cv2.imwrite(fname.name, beamimg2)
        if varimg.get() == True and varbar.get() == False and varintensity.get() == True:
            cv2.imwrite(fname.name, saveimg_intensity)
        if varimg.get() == True and varbar.get() == False and varintensity.get() == False:
            cv2.imwrite(fname.name, saveimg_color)
        if vargray.get() == True:
            cv2.imwrite("%s_gray.%s" % (filename[0],filename[1]), saveimg_gray)
        if varraw.get() == True:
            np.savetxt("%s_RAW.csv" % filename[0], img, delimiter=",")
        if varnormal.get() == True:
            np.savetxt("%s_normalized.csv" % filename[0], img_norm, delimiter=",")
        
    def acsavefile():
        global fname
        fname = tkfd.asksaveasfile(confirmoverwrite=False, defaultextension=".png", filetypes=[("PNG files",".png"),("JPG files",".jpg"),("BMP files",".bmp"),("TIFF files",".tiff")])
        fnamebox.insert(tk.END,fname.name)
        txtname = fname.name.split(".")
        cv2.imwrite(fname.name, acdata2)
        #np.savetxt("%s.csv" % txtname[0], img, delimiter=",")
        try:
            cv2.imwrite("%s_second%s.png" % txtname[0], acdata4)
        except:
            pass
        txtname = fname.name.split(".")
        np.savetxt("%s.csv" % txtname[0], acdata1, delimiter=",")
        
    def set_cam_res():
        global cam_res_w, cam_res_h, cam_res
        #cam.release()
        cam_res = cam_res_box.get()
        cam_res = cam_res.split("x")
        cam_res_w = int(cam_res[0])
        cam_res_h = int(cam_res[1])
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, cam_res_w)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_res_h)
        #ret, frame = cam.read()
        time.sleep(1)
        
    def set_actualsize():
        state = var.get()
        
        return state
    
    def switch_state():
        state = var.get()
        state += 1
        if state == 3:
            state = 0 
        
    def set_gaussian():
        global func
        func = "gaussian"
        print(func)
                
    def set_lorentz():
        global func
        func = "lorentz"
        print(func)
        
    def menu_quit():
        cam.release()
        root.destroy()
        #exit()
        
    def switch_cam(camera_id):
        def x():
            GUI.cam_setup_test(camera_id)
        return x

    def get_var():
        global var_list_get
        var_list = [varraw, varnormal, varimg, vargray, varbar, varintensity]
        for i,j in zip(np.arange(0, len(var_list_get), 1), var_list):
            var_list_get[i] = j.get()

if __name__ == "__main__":
    #sys.modules[__name__].__dict__.clear()
    GUI.cam_setup_test(0)
    GUI.setup()
    #Settings_tab.createtab()
    shutterspeed = 0
    pix1, pix2 = 0, 0
    knifeedge_count = 0
    var = tk.IntVar()
    var.set(0)
    #GUI.cam_setup_test()
    GUI_menu.mainmenu()
    GUI.autocorrelator_graph()
    dark, trackingon = 0, 0
    #root.after(0, GUI.colorbar)
    _, barimg = GUI.colorbar()
    root.after(0, GUI.beamprofiler_img)
    root.after(0, GUI.plotter)
    root.after(0, GUI.beam_width)
    #TAB.createtab(master=testcanvas)
    #TAB.Autocorrelator_tab()
    #TAB.buttons()
    root.mainloop()