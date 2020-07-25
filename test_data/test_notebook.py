# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 03:33:07 2019

@author: keigo
"""

import tkinter as tk
import tkinter.ttk as ttk


if __name__ == "__main__":
    root = tk.Tk()
    root.title("BeamProfiler")
    root.geometry("600x400")
    
    mainframe = ttk.Frame(root, height=400, width=300)
    mainframe.grid(row=1, column=1, rowspan=2, sticky="nswe")
    
    subframe1 = ttk.Notebook(root, width=300, height=200)
    subframe1.grid(row=1, column=2, sticky="nswe")
    
    subframe2 = ttk.Notebook(root, width=300, height=200)
    subframe2.grid(row=2, column=2, sticky="nswe")
    
    frame11 = ttk.Frame(subframe1, height=400, width=300)
    subframe1.add(frame11,text="11")
    
    frame12 = ttk.Frame(subframe1, height=400, width=300)
    subframe1.add(frame12,text="12")
    
    frame21 = ttk.Frame(subframe2, height=400, width=300)
    subframe2.add(frame21,text="21")
    
    frame22 = ttk.Frame(subframe2, height=400, width=300)
    subframe2.add(frame22,text="22")
    
    root.grid_columnconfigure(0, weight = 1, uniform="x")
    root.grid_columnconfigure(1, weight = 1, uniform="x")
    
    #subframe1.grid_propagate(False)
    #subframe2.grid_propagate(False)
    
    def conf(event):
        subframe1.config(height=root.winfo_height(),width=root.winfo_width()-int(root.winfo_width()/2))
        #subframe2.config(height=root.winfo_height()-300,width=root.winfo_width()-300)
        #mainframe.config(height=root.winfo_height()-300,width=root.winfo_width()-300)
    
    def hide_tab(event):
        #print(root.winfo_width())
        if root.winfo_width() <= 600:
            subframe1.grid_forget()
            subframe2.grid_forget()
            #subframe1 = ttk.Notebook(root, width=300, height=200)
            subframe1.grid(row=2, column=1, sticky="nswe")
        if root.winfo_width() > 600:
            subframe1.grid(row=1, column=2, sticky="nswe")
            subframe2.grid(row=2, column=2, sticky="nswe")
            
        #subframe1.config(height=root.winfo_height()-300,width=root.winfo_width()-300)
            
    root.bind("<Configure>",hide_tab)
    root.bind("<Configure>",conf,"+")
    
    
    root.mainloop()