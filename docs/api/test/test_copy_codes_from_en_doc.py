# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import importlib
import json

from parameterized import parameterized

from copy_codes_from_en_doc import (
    load_api_info,
    read_rst_lines_and_copy_info,
    instert_codes_into_cn_rst_if_need,
)


class CopyCodesTest(unittest.TestCase):
    def write_api_info_all(
        self, module_to_use, api_to_use, api_name, api_info_all
    ):
        module_imported = importlib.import_module(module_to_use)
        api_imported = getattr(module_imported, api_to_use)
        with open(api_info_all, 'w') as f:
            json.dump(
                {
                    "123": {
                        "all_names": [
                            api_name,
                        ],
                        "id": 123,
                        "type": "function",
                        "docstring": api_imported.__doc__,
                    }
                },
                f,
            )

    def recovery_rst(self, cnrstfilename, rst_lines):
        with open(cnrstfilename, 'w') as f:
            f.writelines(rst_lines)

    def get_rst_lines(
        self, api_info_all, cnrstfilename_source, cnrstfilename_target
    ):

        # test funcs from copy_codes_from_en_doc
        load_api_info(api_info_all)
        rst_lines_raw, _ = read_rst_lines_and_copy_info(cnrstfilename_source)
        instert_codes_into_cn_rst_if_need(cnrstfilename_source)

        # recovery rst for test
        rst_lines_new, _ = read_rst_lines_and_copy_info(cnrstfilename_source)
        self.recovery_rst(cnrstfilename_source, rst_lines_raw)

        # read target rst to compare
        rst_lines_target, _ = read_rst_lines_and_copy_info(cnrstfilename_target)

        return rst_lines_new, rst_lines_target

    @parameterized.expand(
        [
            (
                'test.case_1_Metric',
                'Metric',
                'paddle.metric.Metric',
                './test/case_1_Metric_api_info_all.json',
                './test/case_1_Metric_cn.rst.source',
                './test/case_1_Metric_cn.rst.target',
            ),
        ]
    )
    def test_copy_from_docstring(
        self,
        module_to_use,
        api_to_use,
        api_name,
        api_info_all,
        cnrstfilename_source,
        cnrstfilename_target,
    ):
        """Test new rst works, which only has COPY-FROM.

        case_1_xxx_api_info_all.json : api info dict
        case_1_xxx.rst.source : original rst file only has COPY-FROM
        case_1_xxx.rst.target : generated rst file converts the COPY-FROM to code-block according to case_1_xxx_api_info_all.json
        """
        self.write_api_info_all(
            module_to_use, api_to_use, api_name, api_info_all
        )

        rst_lines_new, rst_lines_target = self.get_rst_lines(
            api_info_all, cnrstfilename_source, cnrstfilename_target
        )

        assert 'COPY-FROM' not in ' '.join(rst_lines_new)
        assert ' '.join(rst_lines_new) == ' '.join(rst_lines_target)


if __name__ == '__main__':
    unittest.main()
