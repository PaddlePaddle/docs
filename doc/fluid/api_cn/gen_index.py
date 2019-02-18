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


''')
    file_object.write('    ../api_guides/index.rst'+'\n')
    file_object.write('    fluid_cn.rst'+'\n')
    for file_name in sorted(glob.glob("*.rst")):
        if file_name != ('index_cn.rst' and 'fluid_cn.rst'):
            file_object.write('    '+file_name + "\n")
    file_object.close( )
