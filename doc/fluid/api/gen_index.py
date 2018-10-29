import os.path, time
import exceptions
import glob
import os
if __name__ == '__main__':
    
    file_object = open('index.rst', 'w')
    file_object.write('''=============
API Reference
=============

..  toctree::
    :maxdepth: 1

''')
    file_object.write('    api_guides/index.rst'+'\n')
    file_object.write('    fluid.rst'+'\n')
    for file_name in sorted(glob.glob("*.rst")):
        if file_name != ('index_en.rst' and 'fluid.rst'):
            file_object.write('    '+file_name + "\n")
    file_object.close( )
