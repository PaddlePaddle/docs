#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:41:37 2019

@author: v_wanghao11
"""

import io
import re


api_file_name=raw_input("path and name of api_cn.rst to analyse:")

api_file=io.open(api_file_name,'r', encoding='utf-8')
all_api=api_file.read()


all_single_apis=all_api.split(".. _cn_api")
all_single_apis.pop(0)

result_file=io.open("argument_check_.txt",'w',encoding='utf-8')



counter=0  #==>for test, control the api number we want to show when debugging
 #this scripts



for single_api in all_single_apis:
    counter+=1
    if True:
        
        
        
        
        
        #print(single_api)
        lines=single_api.split('\n')
        
        
        '''
        line_index=-1
        for line in lines:
            
            line_index+=1
            
            pattern=re.compile(u"\u53C2\u6570(\uFF1A|:)")
            
            doc_arguments=[]
            if pattern.search(line):
                arg_lines_index=line_index+1
                while arg_lines_index<len(lines) and \
                     (lines[arg_lines_index].startswith(' ') or \
                      len(lines[arg_lines_index])==0) :
                         
                           arg_line=lines[arg_lines_index]
                           arg_line=arg_line.encode('utf-8')
                           print(arg_line)
                           
                           arg_doc_list=arg_line.split("**")
                           if len(arg_doc_list)>2:
                               arg_doc_name=arg_doc_list[1]
                               doc_arguments.append(arg_doc_name)
                          
                           

                           
                           arg_lines_index+=1
                      
            print(doc_arguments)    
        '''
        
        
        
        arguments=[]
        line_index=-1
        for line in lines:
            
        
            line_index+=1
            
            if line.startswith(".. py:function") or\
               line.startswith(".. py:method") or \
               line.startswith(".. py:class") :
                   
                    print("\n\n------"+line+"----------\n\n")
                    
                    #
                    #get arguments in blue block
                    #
                    argument_str=line[line.find('(')+1:line.find(')')]
     
                       
                    #print(argument_str)
                    if len(argument_str)>=1:
                        arguments=argument_str.split(',')
                    #print(arguments)
                    print(argument_str + "  ----block" )
                    
                    
                    
                    doc_arguments=[]
                    for i in range (line_index, len(lines)):
                        
                        # another py:method
                        if i>line_index and lines[i].startswith(".. py:method"):
                            #print("new method begins")
                            break
                        
                        
                        pattern=re.compile(u"\u53C2\u6570(\uFF1A|:)")#参数
                
                        
                        if pattern.search(lines[i]): #参数：这一行
                            #第一行参数，可能为空行，除了空行必须有缩进
                            arg_lines_index=i+1
                            while arg_lines_index<len(lines) and \
                                 (lines[arg_lines_index].startswith(' ') or 
                                  lines[arg_lines_index].startswith('	') or\
                                  len(lines[arg_lines_index])==0) :
                                       '''
                                       if i>arg_lines_index and lines[i].startswith(".. py:method"):
                                               print("new method begins")
                                               break
                                       '''
                                       arg_line=lines[arg_lines_index]
                                       arg_line=arg_line.encode('utf-8')
                                       print(arg_line)
                                       
                                       arg_doc_list=arg_line.split("**")
                                       if len(arg_doc_list)>2:
                                           arg_doc_name=arg_doc_list[1]
                                           doc_arguments.append(arg_doc_name)
                                      
                                       
            
                                       
                                       arg_lines_index+=1
                                       
                            print(str(doc_arguments) + "  ----doc")
                            
                    
                    
                    ok=0
                    if len(arguments)!=len(doc_arguments):
                        ok=1
                        
                        
                    else:
                        for i in range(0,len(arguments)):
                            if re.search(doc_arguments[i],arguments[i]):
                                continue
                            else:
                                ok=2
                                
                    if ok!=0:
                        result_file.writelines("\n\n-----------"+line+"----------\n\n")
                        result_file.write(u'['+argument_str + u"] --- in blue block\n")
                        
                        doc_arguments_str=str(doc_arguments)
                        doc_arguments_str=doc_arguments_str.decode('utf-8')
                        result_file.write(doc_arguments_str + u"  ----doc\n")
                        
                        if ok==1:
                            print("!!!!!!!!参数长度不匹配")
                            result_file.write(u"!!!!!!!!参数长度不匹配\n")
                        if ok==2:
                            print("!!!!!!!!参数不一一对应，"+doc_arguments[i])
                            result_file.write(u"!!!!!!!!参数不一一对应"+ \
                                                  str(doc_arguments[i]).decode('utf-8')+ \
                                                  u"\n") 
        
    

result_file.close()
api_file.close()