import os.path, time
import exceptions
import glob
import os
if __name__ == '__main__':
    
    file_object = open('index_cn.rst', 'w')
    file_object.write('''=============
API Reference
=============

..  toctree::
    :maxdepth: 1

''')
    file_object.write('    ../api_guides/index_cn.rst'+'\n')

    file_names = []
    file_names = glob.glob("*.rst")
    
    for file_name in sorted(file_names):
        with open(file_name, 'r')as f:
            for i in range(2):
                line = f.readline().strip()
                if line.find('paddle.') != -1:
                    file_object.write('    '+file_name + "\n")
                    file_names.remove(file_name)

    file_object.write('    fluid_cn.rst'+'\n')
    for file_name in sorted(file_names):
        if file_name != 'index.rst' and file_name != 'index_cn.rst' and file_name != 'fluid_cn.rst':
            file_object.write('    '+file_name + "\n")
    file_object.close( )
