import sys


def check_copy_from_not_parsed(file):
    error_parsed = []
    with open(file, 'r') as f:
        for line, i in enumerate(f):
            if 'COPY-FROM' in i:
                error_parsed.append(i)
                print(
                    "ERROR: ",
                    file,
                    "line: ",
                    line + 1,
                    i,
                    " is not parsed into sample code, \
                    please check the api name after COPY-FROM",
                )
    return error_parsed


def run_copy_from_check(output_path, pr_files):
    print('COPY-FROM check files: ', pr_files, ' in ', output_path)
    all_error_parsed = []
    if not pr_files:
        print("pr file list is empty, skip COPY-FROM check")
        sys.exit(0)
    for file in pr_files:
        if '_cn.rst' not in file:
            return error_parsed
        # find the Chinese HTML file for PR File
        error_parsed = check_copy_from_not_parsed(
            output_path + file.replace('_cn.rst', '_cn.html')
        )
        all_error_parsed.extend(error_parsed)
    if all_error_parsed:
        sys.exit(1)
    print("All COPY-FROM parsed success in PR !")


if len(sys.argv) < 2:
    print("Error: inadequate number of arguments")
    print("Please input one file path")
    sys.exit(1)
else:
    res = run_copy_from_check(sys.argv[1], sys.argv[2:])
