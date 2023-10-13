import sys
import os
import re

# check file's api_label
def check_api_label(rootdir, file):
    real_file = rootdir + file
    with open(real_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    if first_line == en_label_creater(file):
        return True
    return False


# path -> api_label(the fist line 's style)
def en_label_creater(file):
    result = re.sub("api/", "", file)
    result = re.sub("_cn.rst", "", result)
    result = re.sub('/', "_", result)
    result = '.. _cn_' + result + ':'
    return result


# traverse doc/api to append api_label in list
def traverse_api_label(rootdir):
    list = []
    for root, dirs, files in os.walk(rootdir + 'api/'):
        for file in files:
            real_path = os.path.join(root, file)
            path = re.sub(rootdir, "", real_path)
            if test_ornot(path):
                for label in get_api_label_list(real_path):
                    list.append(label)
    return list


# api_labels in a file
def get_api_label_list(file_path):
    list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(".. _cn"):
                line = re.sub(".. _", "", line)
                line = re.sub(":", "", line)
                list.append(line.rstrip())
    return list


# api doc for checking
def test_ornot(file):
    if (
        file.endswith("_cn.rst")
        and (file not in ["Overview_cn.rst", "index_cn.rst"])
        and file.startswith("api")
    ):
        return True
    return False


def pipline(rootdir, files):
    for file in files:
        if test_ornot(file):
            if check_api_label(rootdir, file):
                pass
            else:
                print("error:", file)
    list = traverse_api_label(rootdir)
    for file in files:
        with open(rootdir + file, 'r', encoding='utf-8') as f:
            pattern = f.read()
        matches = re.findall(r":ref:`([^`]+)`", pattern)
        if matches:
            for match in matches:
                if match.startswith('cn'):
                    if match not in list:
                        print(rootdir + file, 'ref' + match, 'error')
                else:
                    out = re.search("<(.*?)>", match)
                    if out and out.group(1).startswith("cn"):
                        if out.group(1) not in list:
                            print(rootdir + file, 'ref' + match, 'error')


if __name__ == "__main__":
    pipline(sys.argv[1], sys.argv[2:])
