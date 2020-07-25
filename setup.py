# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:57:11 2019

@author: user
"""

# coding: utf-8

############### cx_Freeze 用セットアップファイル ##########
# コマンドライン上で python setup.py buildとすると、exe化　#
# Mac用のAppを作成するには、buildをbdist_macとする        #
######################################################
 
import sys, os
from cx_Freeze import setup, Executable

#個人的な設定（コマンドライン上でファイルをぶっこみたい）
file_path = input("アプリ化したいpy：")

#TCL, TKライブラリのエラーが発生する場合に備え、以下を設定
#参考サイト：http://oregengo.hatenablog.com/entry/2016/12/23/205550
if sys.platform == "win32":
    base = "win32GUI" # "Win32GUI" ←GUI有効
    #Windowsの場合の記載　それぞれの環境によってフォルダの数値等は異なる
    os.environ['TCL_LIBRARY'] = "C:\\Users\\keigo\\Anaconda3\\tcl\\tcl8.6"
    os.environ['TK_LIBRARY'] = "C:\\Users\\keigo\\Anaconda3\\tcl\\tk8.6"
else:
    base = None # "Win32GUI"

#importして使っているライブラリを記載
packages = []

#importして使っているライブラリを記載（こちらの方が軽くなるという噂）
includes = [
    "tkinter",
    "numpy",
    "cv2",
    "pandas",
    "matplotlib",
    "scipy",
    "time",
    "PIL",
    "platform",
    "win32api",
]

#excludesでは、パッケージ化しないライブラリやモジュールを指定する。
"""
numpy,pandas,lxmlは非常に重いので使わないなら、除く。（合計で80MBほど）
他にも、PIL(5MB)など。
"""
excludes = ['boto.compat.sys',
            'boto.compat._sre',
            'boto.compat._json',
            'boto.compat._locale',
            'boto.compat._struct',
            'boto.compat.array',]

##### 細かい設定はここまで #####

#アプリ化したい pythonファイルの指定触る必要はない
exe = Executable(
    script = file_path,
    base = base,
    icon = "icon.png"
)
 
# セットアップ
setup(name = 'main',
      options = {
          "build_exe": {
              "packages": packages, 
              "includes": includes, 
              "excludes": excludes,
          }
      },
      version = '0.10.5',
      description = 'converter',
      executables = [exe])