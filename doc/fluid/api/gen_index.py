import os.path, time
import exceptions
import glob
import os

if __name__ == '__main__':
    with open('index_en.rst', 'w') as file_object:
        file_object = open('index_en.rst', 'w')
        file_object.write('''=============
API Reference
=============

..  toctree::
    :maxdepth: 1

    ../api_guides/index_en.rst
''')

        target_dirs = ['.', 'data']

        file_names = []
        for target_dir in target_dirs:
            if target_dir == '.':
                pattern = '*.rst'
            else:
                pattern = target_dir + '/*.rst'
            file_names.extend(glob.glob(pattern))

        for file_name in sorted(file_names):
            with open(file_name, 'r') as f:
                for i in range(2):
                    line = f.readline().strip()
                    if line.find('paddle.') != -1:
                        file_object.write('    ' + file_name + "\n")
                        file_names.remove(file_name)

        file_object.write('    ' + 'fluid.rst' + "\n")
        for file_name in sorted(file_names):
            if file_name not in ['index_en.rst']:
                file_object.write('    ' + file_name + "\n")
