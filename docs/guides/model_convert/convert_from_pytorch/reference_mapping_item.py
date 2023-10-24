import os
import re
import typing

whitelist = set(['torch.Tensor.requires_grad_.md'])


class DiffMeta(typing.TypedDict):
    torch_api: str
    torch_api_url: typing.Optional[str]
    paddle_api: typing.Optional[str]
    paddle_api_url: typing.Optional[str]
    mapping_type: str
    source_file: str


def getMetaFromDiffFile(filepath):
    meta_data: DiffMeta = {'source_file': filepath}
    state = 0
    # 0: wait for title
    # 1: wait for torch api
    # 2: wait for paddle api
    # 3: end
    title_pattern = re.compile(r"^## +\[(?P<type>[^\]]+)\] *(?P<torch_api>.+)$")
    torch_pattern = re.compile(
        r"^### +\[ *(?P<torch_api>torch.[^\]]+)\](?P<url>\([^\)]*\))?$"
    )
    paddle_pattern = re.compile(
        r"^### +\[ *(?P<paddle_api>paddle.[^\]]+)\](?P<url>\([^\)]*\))$"
    )

    with open(filepath, 'r') as f:
        for line in f.readlines():
            if not line.startswith('##'):
                continue

            if state == 0:
                title_match = title_pattern.match(line)
                if title_match:
                    mapping_type = title_match['type'].strip()
                    torch_api = title_match['torch_api'].strip()

                    meta_data['torch_api'] = torch_api
                    meta_data['mapping_type'] = mapping_type
                    state = 1
                else:
                    raise Exception(f"Cannot parse title: {line} in {filepath}")
            elif state == 1:
                torch_match = torch_pattern.match(line)

                if torch_match:
                    torch_api = torch_match['torch_api'].strip()
                    torch_url = torch_match['url'] if torch_match['url'] else ''
                    real_url = torch_url.lstrip('(').rstrip(')')
                    if meta_data['torch_api'] != torch_api:
                        raise Exception(
                            f"torch api not match: {line} != {meta_data['torch_api']} in {filepath}"
                        )
                    meta_data['torch_api_url'] = real_url
                    state = 2
                else:
                    raise Exception(
                        f"Cannot parse torch api: {line} in {filepath}"
                    )
            elif state == 2:
                paddle_match = paddle_pattern.match(line)

                if paddle_match:
                    paddle_api = paddle_match['paddle_api'].strip()
                    paddle_url = paddle_match['url'].strip()
                    real_url = paddle_url.lstrip('(').rstrip(')')
                    meta_data['paddle_api'] = paddle_api
                    meta_data['paddle_api_url'] = real_url
                    state = 3
            else:
                pass

    if state < 2:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {filepath}, current meta: {meta_data}"
        )

    return meta_data


REFERENCE_PATTERN = re.compile(
    r'^\| *REFERENCE-MAPPING-ITEM\( *(?P<torch_api>[^,]+) *, *(?P<diff_url>.+) *\) *\|$'
)


def apply_reference_to_row(line, metadata_dict, table_row_idx, line_idx):
    reference_match = REFERENCE_PATTERN.match(line)
    if reference_match:
        torch_api = reference_match['torch_api'].strip('`')
        diff_url = reference_match['diff_url']

        row_idx_s = str(table_row_idx)

        reference_item = metadata_dict.get(torch_api, None)
        torch_api_url = reference_item['torch_api_url']
        torch_api_column = f'[`{torch_api}`]({torch_api_url})'

        if 'paddle_api' not in reference_item:
            print(
                f"Cannot find paddle_api for torch_api: {torch_api} in line {line_idx}"
            )
            paddle_api_column = ''
        else:
            paddle_api = reference_item['paddle_api']
            paddle_api_url = reference_item['paddle_api_url']
            paddle_api_column = f'[`{paddle_api}`]({paddle_api_url})'

        mapping_type = reference_item['mapping_type']
        mapping_column = f'{mapping_type}，[差异对比]({diff_url})'

        content = [
            row_idx_s,
            torch_api_column,
            paddle_api_column,
            mapping_column,
        ]

        output = '| ' + ' | '.join(content) + ' |\n'
        return output
    else:
        return line


def reference_mapping_item(index_path, metadata_dict):
    if not os.path.exists(index_path):
        raise Exception(f"Cannot find pytorch_api_mapping_cn.md: {index_path}")

    with open(mapping_index_file, "r", encoding='utf-8') as f:
        lines = f.readlines()

    state = 0
    # -1: error
    # 0: wait for table header

    # 1: wait for ignore table seperator
    # 2: wait for expect table content

    # 5: wait for ignore table content
    # 6: wait for expect table content

    column_names = []
    column_count = -1
    table_seperator_pattern = re.compile(r"^ *\|(?P<group> *-+ *\|)+ *$")

    expect_column_names = ['序号', 'PyTorch API', 'PaddlePaddle API', '备注']

    table_row_idx = -1
    output = []

    for i, line in enumerate(lines):
        if state < 0:
            break

        content = line.strip()
        if not content.startswith('|'):
            output.append(line)
            state = 0
            continue

        columns = [c.strip() for c in content.split('|')]
        if len(columns) <= 2:
            raise Exception(
                f'Table column count must > 0, but found {len(columns) - 2} at line {i+1}: {line}'
            )
        columns = columns[1:-1]

        if state == 0:
            column_names.clear()
            column_names.extend([c.strip() for c in columns])
            column_count = len(column_names)
            if column_names == expect_column_names:
                state = 2
                table_row_idx = 1
                # print(f'process mapping table at line {i+1}.')
            else:
                state = 1
                print(f'ignore table with {column_names} at line {i+1}.')
            output.append(line)
        elif state == 1:
            if (
                not table_seperator_pattern.match(line)
                or len(columns) != column_count
            ):
                raise Exception(
                    f"Table seperator not match at line {i+1}: {line}"
                )
            state = 5
            output.append(line)
        elif state == 2:
            if (
                not table_seperator_pattern.match(line)
                or len(columns) != column_count
            ):
                raise Exception(
                    f"Table seperator not match at line {i+1}: {line}"
                )
            state = 6
            output.append(line)
        elif state == 5:
            # if len(columns) != column_count:
            #     raise Exception(
            #         f"Table content not match at line {i+1}: {line}"
            #     )
            output.append(line)
            # state = 5
        elif state == 6:
            # if len(columns) != column_count:
            #     raise Exception(
            #         f"Table content not match at line {i+1}: {line}"
            #     )
            try:
                referenced_row = apply_reference_to_row(
                    line, metadata_dict, table_row_idx, i + 1
                )
                table_row_idx += 1

                output.append(referenced_row)
            except Exception as e:
                print(e)
                print(f"Error at line {i+1}: {line}")
                output.append(line)

            # state = 6
        else:
            raise Exception(
                f"Unexpected State at {state} in processing file: {index_path}"
            )

    if state == 5 or state == 6:
        state = 0

    if state != 0:
        raise Exception(
            f"Unexpected End State at {state} in parsing file: {index_path}"
        )

    with open(mapping_index_file, "w", encoding='utf-8') as f:
        f.writelines(output)


if __name__ == '__main__':
    # convert from pytorch basedir
    cfp_basedir = os.path.dirname(__file__)
    # pytorch_api_mapping_cn
    mapping_index_file = os.path.join(cfp_basedir, 'pytorch_api_mapping_cn.md')

    api_difference_basedir = os.path.join(cfp_basedir, 'api_difference')

    mapping_file_pattern = re.compile(r"^torch\.(?P<api_name>.+)\.md$")
    # get all diff files (torch.*.md)
    diff_files = sorted(
        [
            os.path.join(path, filename)
            for path, _, file_list in os.walk(api_difference_basedir)
            for filename in file_list
            if mapping_file_pattern.match(filename)
            and filename not in whitelist
        ]
    )

    metas = [getMetaFromDiffFile(f) for f in diff_files]

    meta_dict = {m['torch_api']: m for m in metas}

    output = reference_mapping_item(mapping_index_file, meta_dict)
