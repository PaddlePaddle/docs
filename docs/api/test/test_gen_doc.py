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
import json
from parameterized import parameterized

from gen_doc import extract_code_blocks_from_docstr


class ExtractTest(unittest.TestCase):
    def load_api_info(self, api_info_all):
        api_name_2_id_map = {}
        with open(api_info_all, 'r') as f:
            api_info_dict = json.load(f)
        for k, api_info in api_info_dict.items():
            for n in api_info.get('all_names', []):
                api_name_2_id_map[n] = k

        return api_info_dict, api_name_2_id_map

    @parameterized.expand(
        [
            (
                './test/case_0_clip_RecordEvent_cn_api_info_all.json',
                'paddle.profiler.RecordEvent',
                True,
                1,
                [True],
            ),
            (
                './test/case_0_clip_RecordEvent_cn_api_info_all.json',
                'paddle.profiler.RecordEvent',
                False,
                1,
                [True],
            ),
            (
                './test/case_0_gather_cn_api_info_all.json',
                'paddle.gather',
                True,
                1,
                [True],
            ),
            (
                './test/case_0_gather_cn_api_info_all.json',
                'paddle.gather',
                False,
                2,
                [False, True],
            ),
            (
                './test/case_1_clip_gather_cn_api_info_all.json',
                'paddle.gather',
                True,
                0,
                [],
            ),
            (
                './test/case_1_clip_gather_cn_api_info_all.json',
                'paddle.gather',
                False,
                1,
                [False],
            ),
        ]
    )
    def test_extract(
        self, api_info_all, api_key, google_style, len_code_block, in_examples
    ):
        api_info_dict, api_name_2_id_map = self.load_api_info(api_info_all)
        api_info = api_info_dict[api_name_2_id_map[api_key]]

        code_blocks = extract_code_blocks_from_docstr(
            api_info['docstring'], google_style=google_style
        )

        assert len(code_blocks) == len_code_block

        for idx, cb in enumerate(code_blocks):
            assert in_examples[idx] == cb['in_examples']


if __name__ == '__main__':
    unittest.main()
