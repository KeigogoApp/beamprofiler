# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:33:38 2019

@author: keigo
"""

import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

var1 = config["DEFAULT"]["user"]
var2 = config["DEFAULT"]["age"]

config.set("DEFAULT", "test", "ok")
var3 = config["DEFAULT"]["test"]

config.set("DEFAULT", "age", "18")
var4 = config["DEFAULT"]["age"]

print(var1)
print(var2)
print(var3)
print(var4)

with open('config.ini', 'w') as file:
    config.write(file)
    