import math
import os
import pickle
import shutil
import subprocess
import multiprocessing
import sys


def removeSomeApis(filenames):
    filenames.remove('./fluid_cn/DistributeTranspiler_cn.rst')
    filenames.remove('./transpiler_cn/DistributeTranspiler_cn.rst')
    filenames.remove('./transpiler_cn/DistributeTranspilerConfig_cn.rst')
    filenames.remove('./transpiler_cn/HashName_cn.rst')
    filenames.remove('./transpiler_cn/memory_optimize_cn.rst')
    filenames.remove('./transpiler_cn/release_memory_cn.rst')
    filenames.remove('./transpiler_cn/RoundRobin_cn.rst')
    for i in range(len(filenames)-1, -1, -1):
        length = len(filenames[i].split("/"))
        if length == 2:
            filenames.pop(i)
    return filenames


def find_all(src_str, substr):
    indices = []
    get_one = src_str.find(substr)
    while get_one != -1:
        indices.append(get_one)
        get_one = src_str.find(substr, get_one + 1)
    return indices


def extract_sample_code(srcfile, status_all):
    filename = srcfile.name
    srcc = srcfile.read()
    srcfile.seek(0, 0)
    srcls = srcfile.readlines()
    status = []
    sample_code_begins = find_all(srcc, " code-block:: python")
    if len(sample_code_begins) == 0:
        status.append(-1)

    else:
        for i in range(0, len(srcls)):
            if srcls[i].find(".. code-block:: python") != -1:
                content = ""
                start = i
                startindent = srcls[start + 2][:srcls[start + 2].find("import")]
                content += srcls[start + 2][len(startindent):]
                for j in range(start + 3, len(srcls)):
                    if not srcls[j].startswith(startindent) and srcls[j] != '\n':
                        break
                    if srcls[j].find(" code-block:: python") != -1:
                        break
                    content += srcls[j].replace(startindent, "", 1)
                status.append(run_sample_code(content, filename))

    status_all[filename] = status
    return status_all


def run_sample_code(content, filename):
    fname = filename.split("/")[-1].replace("_cn", "").replace(".rst", "") + ".py"
    tempf = open("temp/" + fname, 'w')
    tempf.write(content)
    tempf.close()
    cmd = ["python", "temp/" + fname]

    subprc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, error = subprc.communicate()
    err = "".join(error.decode(encoding='utf-8'))

    if subprc.returncode != 0:
        print("\nSample code error found in ", filename, ":\n")
        print("subprocess return code: ", str(subprc.returncode))
        print("Error Raised from Sample Code  content:\n", content, " :\n")
        print(err)
        status = 1
    else:
        status = 0
    os.remove("temp/" + fname)
    return status

def test(file):
    temp = []
    src = open(file, 'r')
    status_all = {}
    extract_sample_code(src, status_all)
    temp.append(status_all)
    src.close()
    return temp


if not os.path.isdir("temp"):
    os.mkdir("temp")

output = []

if len(sys.argv) < 2:
    print("Error: inadequate number of arguments")
    print("Please one file")
    sys.exit(1)
else:
    if not os.path.exists(sys.argv[1]):
        print("File not found")
        sys.exit(1)
    res = test(sys.argv[1])    
    output.append(res)


status_groups = {-1: [], 0: [], 1: []}
ci_pass = True
for one_file in output:
    for dicts in one_file:
        for key in dicts:
            status = dicts[key]
            for ele in status:
                if ele != 0:
                    ci_pass = False
                    break
            if len(status) == 1:
                status_groups[status[0]].append(key)
            else:
                for u in range(0, len(status)):
                    status_groups[status[u]].append(key + '_' + str(u + 1))

error_api = status_groups[-1] + status_groups[1]
total_error_number = len(error_api)


print("****************************************************")
print("----------------End of the Check--------------------")
print("****************************************************")
if total_error_number > 0:
    print("Error sample code number is:{}".format(total_error_number))
    type_one_number = len(status_groups[-1])
    type_two_number = len(status_groups[1])
    if type_one_number > 0:
        print("Error type one sample number is:{}".format(type_one_number))
        print("Error raised from type one:no sample code.", str(status_groups[-1]))
    if type_two_number > 0:
        print("Error type two sample number is:{}".format(type_two_number))
        print("Error raised from type two:running error sample code.", str(status_groups[1]))
if not ci_pass:
    print("Mistakes found in sample codes.")
    exit(1)
else:
    print("Sample code check is successful!")
