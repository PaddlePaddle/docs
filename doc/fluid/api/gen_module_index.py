import glob 
import sys
import os

def print_module_index(module, header):
    modules = module.split('.')
    if len(modules) > 1:
        os.chdir('/'.join(modules[0:-1]))
        pattern = modules[-1] + '/*.rst'
        stream = open(modules[-1] + '.rst', 'w')
    else:
        pattern = modules[0] + '/*.rst'
        stream = open(modules[0] + '.rst', 'w')

    stream.write('=' * len(header) + '\n')
    stream.write(header + '\n')
    stream.write('=' * len(header) + '\n')
    stream.write('''
..  toctree::
    :maxdepth: 1

''')

    blank_num = 4 
    files = sorted(glob.glob(pattern), key=str.lower)

    for f in files:
        if f == "io/PipeReader.rst": 
            continue
        stream.write(' ' * blank_num)
        stream.write(f)
        stream.write('\n')

    stream.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python gen_module_index.py [module_name] [header_name]')
        sys.exit(-1)

    print_module_index(sys.argv[1], sys.argv[2])
