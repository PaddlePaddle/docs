#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:02:10 2019

@author: v_wanghao11
"""

import os
import subprocess

root_dir="./api_codes"
file_list=[]
for root,dirs,file_name in os.walk(root_dir):
    file_list+=file_name

#print(file_list)

report=open("report.txt",'w')

for file_name in file_list:
    
    report.write("\n\nFileName:"+file_name+":\n")
    report.write("---------------------------\n\n")
    
    cmd=["python" , root_dir+"/"+file_name]
    
    subprc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = subprc.communicate()
    for msg in output:
        report.write(msg)
    

report.close();