import re
import math
import os
import pickle
import shutil
import subprocess
import multiprocessing
import sys


def remove_desc_code(srcls, filename):
    if filename == 'fluid_cn/one_hot_cn.rst':
        srcls.pop(13)
        srcls.pop(28)
        srcls.pop(44)
    if filename == 'layers_cn/one_hot_cn.rst':
        srcls.pop(15)
        srcls.pop(30)
        srcls.pop(46)
    if filename == 'profiler_cn/profiler_cn.rst':
        srcls.pop(41)
    if filename == 'layers_cn/natural_exp_decay_cn.rst':
        srcls.pop(13)
    if filename == 'layers_cn/transpose_cn.rst':
        srcls.pop(20)
    if filename == 'layers_cn/array_length_cn.rst':
        srcls.pop(36)
    if filename == 'layers_cn/inverse_time_decay_cn.rst':
        srcls.pop(13)
    if filename == 'layers_cn/stack_cn.rst':
        srcls.pop(12)
        srcls.pop(33)
    if filename == 'layers_cn/sums_cn.rst':
        srcls.pop(11)
    if filename == 'layers_cn/sum_cn.rst':
        for i in range(len(srcls) - 1, 61, -1):
            srcls.pop(i)
    if filename == 'layers_cn/softmax_cn.rst':
        srcls.pop(30)
        srcls.pop(57)
    if filename == 'layers_cn/array_write_cn.rst':
        srcls.pop(37)
    if filename == 'layers_cn/lod_append_cn.rst':
        srcls.pop(11)
    if filename == 'layers_cn/reorder_lod_tensor_by_rank_cn.rst':
        srcls.pop(25)
    if filename == 'layers_cn/round_cn.rst':
        srcls.pop(10)
    if filename == 'layers_cn/squeeze_cn.rst':
        srcls.pop(11)
        srcls.pop(19)
        srcls.pop(27)
    if filename == 'layers_cn/unsqueeze_cn.rst':
        srcls.pop(11)
    if filename == 'layers_cn/array_read_cn.rst':
        srcls.pop(51)
    if filename == 'layers_cn/scatter_cn.rst':
        srcls.pop(9)
    if filename == 'layers_cn/topk_cn.rst':
        srcls.pop(11)
    if filename == 'optimizer_cn/ModelAverage_cn.rst':
        srcls.pop(15)
    return srcls


def check_indent(code_line):
    indent = ""
    for c in code_line:
        if c == '\t':
            indent += '    '
        elif c == ' ':
            indent += ' '
        if c != ' ' and c != '\t':
            break
    return indent


def find_all(src_str, substr):
    indices = []
    get_one = src_str.find(substr)
    while get_one != -1:
        indices.append(get_one)
        get_one = src_str.find(substr, get_one + 1)
    return indices


def strip_ps1_from_codeblock(codeblock):
    """strip PS1(>>> ) from codeblock"""
    mo = re.search(r"\n>>>\s?", "\n" + codeblock)
    if mo is None:
        return codeblock

    codeblock_clean = []
    for line in codeblock.splitlines():
        mo = re.match(r"^>>>\s?", line.lstrip())
        if mo is None:
            codeblock_clean.append("# {}".format(line))
        else:
            codeblock_clean.append(line[mo.end() :])

    return "\n".join(codeblock_clean)


def extract_sample_code(srcfile, status_all):
    filename = srcfile.name
    srcc = srcfile.read()
    srcfile.seek(0, 0)
    srcls = srcfile.readlines()
    srcls = remove_desc_code(
        srcls, filename
    )  # remove description info for samplecode
    status = []
    sample_code_begins = find_all(srcc, " code-block:: python")
    if len(sample_code_begins) == 0:
        status.append(-1)

    else:
        for i in range(0, len(srcls)):
            if srcls[i].find(".. code-block:: python") != -1:
                content = ""
                start = i

                blank_line = 1
                while srcls[start + blank_line].strip() == '':
                    blank_line += 1

                startindent = ""
                # remove indent error
                if srcls[start + blank_line].find("from") != -1:
                    startindent += srcls[start + blank_line][
                        : srcls[start + blank_line].find("from")
                    ]
                elif srcls[start + blank_line].find("import") != -1:
                    startindent += srcls[start + blank_line][
                        : srcls[start + blank_line].find("import")
                    ]
                else:
                    startindent += check_indent(srcls[start + blank_line])
                content += srcls[start + blank_line][len(startindent) :]
                for j in range(start + blank_line + 1, len(srcls)):
                    # planish a blank line
                    if (
                        not srcls[j].startswith(startindent)
                        and srcls[j] != '\n'
                    ):
                        break
                    if srcls[j].find(" code-block:: python") != -1:
                        break
                    content += srcls[j].replace(startindent, "", 1)
                status.append(
                    run_sample_code(strip_ps1_from_codeblock(content), filename)
                )

    status_all[filename] = status
    return status_all


def run_sample_code(content, filename):
    # three status ,-1:no sample code; 1: running error; 0:normal
    fname = (
        filename.split("/")[-1].replace("_cn", "").replace(".rst", "") + ".py"
    )
    tempf = open("temp/" + fname, 'w')
    content = "# -*- coding: utf-8 -*-\n" + content
    tempf.write(content)
    tempf.close()
    cmd = ["python", "temp/" + fname]

    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, error = subprc.communicate()
    err = "".join(error.decode(encoding='utf-8'))

    if subprc.returncode != 0:
        print("\nSample code error found in ", filename, ":\n")
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


if os.path.isdir("temp"):
    shutil.rmtree("temp")
if os.path.isdir("infer_model"):
    shutil.rmtree("infer_model")
if os.path.isdir("image"):
    shutil.rmtree("image")
if os.path.isdir("my_paddle_model"):
    shutil.rmtree("my_paddle_model")
if os.path.isdir("my_paddle_vars"):
    shutil.rmtree("my_paddle_vars")

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
# polishes show format
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
    if type_two_number > 0:
        print("Error type two sample number is:{}".format(type_two_number))
        print(
            "Error raised from type two:running error sample code.",
            str(status_groups[1]),
        )
    if type_one_number > 0:
        print("Error type one sample number is:{}".format(type_one_number))
        print(
            "Error raised from type one:no sample code.", str(status_groups[-1])
        )
        # no sample code not considered as error now.
        if type_two_number == 0:
            ci_pass = True
if not ci_pass:
    print("Mistakes found in sample codes.")
    exit(1)
else:
    print("Sample code check is successful!")
