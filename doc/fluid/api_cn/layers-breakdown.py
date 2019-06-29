#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:36:04 2019


break down layers_cn to its sub categories

@author: v_wanghao11
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
    '''
    print "lines[i]:"+srclines[i]
    print re.match(r'^=+$', srclines[i])
    '''
    if re.match(r'^=+$', srclines[i])!=None:
        title=""
        base_idx=i+1
        for j in range(base_idx,len(srclines)):# j is real index
            '''
            print srclines[j]
            raw_input("1")
            '''
            if re.match(r'^=+$', srclines[j])!=None:
                '''
                print "title end"
                raw_input("2")
                '''
                title="".join(srclines[base_idx:j])
                title=title.strip().replace('\n','')

                '''
                print "here"+title
                raw_input("3")
                '''
                titles[title]=(i,j)
                i=j+1
                break
        #   raw_input("4")
    else:
        i+=1



print titles

titlines=titles.values()
titlines.sort()

print titlines


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
            
            #

            if i==len(titlines)-1:
                keyf.write("".join(srclines[titlines[i][1]+1:]))
            else:
                
                keyf.write("".join(srclines[titlines[i][1]+1:titlines[i+1][0]]))
            
            keyf.close()
        



