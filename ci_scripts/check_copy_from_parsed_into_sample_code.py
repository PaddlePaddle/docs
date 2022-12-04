import sys
import re
import os


def check_copy_from_not_parsed(file):
    error_parsed = []
    with open(file, 'r') as f:
        for line, i in enumerate(f):
            if 'COPY-FROM' in i:
                error_parsed.append(i)
                print(
                    file,
                    "line: ",
                    line,
                    i,
                    " is not parsed into sample code, \
                    please check the api name after COPY-FROM",
                )
    return error_parsed


def run(input_files):
    print('COPY-FROM check files: ', input_files)
    all_error_parsed = []
    if not input_files:
        print("input_files list is empty, skip COPY-FROM check")
        sys.exit(0)
    for file in input_files:
        error_parsed = check_copy_from_not_parsed("../docs/" + file)
        all_error_parsed.extend(error_parsed)
    if all_error_parsed:
        sys.exit(1)
    print("All COPY-FROM parsed success in PR !")


if len(sys.argv) < 2:
    print("Error: inadequate number of arguments")
    print("Please input one file path")
    sys.exit(1)
else:
    res = run(sys.argv[1:])
