import os.path, time
import exceptions
import glob
import os
if __name__ == '__main__':
    
    file_object = open('index_en.rst', 'w')
    file_object.write('''=============
API Reference
=============

..  toctree::
    :maxdepth: 1

''')
    file_object.write('    ../api_guides/index_en.rst'+'\n')
    #file_object.write('    fluid.rst/index_en.rst'+'\n')
    for file_name in sorted(glob.glob("*.rst")):
        #if file_name != 'index_en.rst' and file_name != 'fluid.rst':
        if file_name != 'index_en.rst':
            for f in os.walk(file_name):
                files_list = f[2] 
                file_obj = file_name.split('.')
                with open('%s/index_en.rst' %file_name, 'w') as file_name_object:
                    if file_name == 'fluid.rst':
                        file_last = ""
                    else:
                        file_last = '.' + file_obj[0]
                    file_name_object.write('''===========
%s
===========

..  toctree::
    :maxdepth: 1
\n''' %("fluid"+file_last) )
   # %s
                   # \n''' %("fluid."+file_obj[0]) )
                    for files in files_list:
                        file_name_object.write('''    %s\n''' %files)
            file_object.write('    ' + file_name + "/" + "index_en.rst" + "\n")
    file_object.close( )
