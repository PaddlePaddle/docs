#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:36:04 2019


break down layers_cn to its sub categories

@author: haowang101779990
"""

import sys
import os
import re

stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys)
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')

srcfile=open("layers_cn.rst",'r')
srclines=srcfile.readlines()
srcfile.close()

titles={}

i=0
while i <len(srclines):

    if re.match(r'^=+$', srclines[i])!=None:

        title=""
        base_idx=i+1

        for j in range(base_idx,len(srclines)):

            if re.match(r'^=+$', srclines[j])!=None:
                title="".join(srclines[base_idx:j])
                title=title.strip().replace('\n','')
                titles[title]=(i,j)
                i=j+1
                break

    else:
        i+=1

titlines=titles.values()
titlines.sort()

if not os.path.isdir("./layers_cn"):
    os.mkdir("layers_cn")

for i in range(0,len(titlines)):
    for key in titles.keys():
        if(titles[key]==titlines[i]):

            keyf=open("layers_cn/"+key+"_cn.rst",'w')
            
            #title for this file
            for _ in range(0,len(key)+5):
                keyf.write("=")
            keyf.write("\n"+key+"\n")
            for _ in range(0,len(key)+5):
                keyf.write("=")
            keyf.write("\n")

            #write into file
            if i==len(titlines)-1:
                keyf.write("".join(srclines[titlines[i][1]+1:]))
            else:     
                keyf.write("".join(srclines[titlines[i][1]+1:titlines[i+1][0]]))
            
            keyf.close()
