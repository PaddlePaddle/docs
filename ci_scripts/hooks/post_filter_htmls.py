#! /bin/env python

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
do sth. after the html generated.
"""
import argparse
import os
import sys

from bs4 import BeautifulSoup


def insert_header_and_anchor_for_method(htmlfile):
    """
    insert a hide h3 tag and a anchor for every class method.
    """
    soup = BeautifulSoup(open(htmlfile, "r"), "lxml")
    method_title_tags = soup.find_all("dl", class_="method")
    for mtt in method_title_tags:
        dt = mtt.find("dt")
        descname_objs = dt.select(
            "code.descname span.pre, span.descname span.pre"
        )
        if len(descname_objs) < 1:
            continue
        method_name = descname_objs[0].text
        new_h3 = soup.new_tag("h3", style="display:none")
        new_h3.string = method_name
        new_anchor = soup.new_tag(
            "a",
            attrs={
                "class": "headerlink",
                "href": "#" + method_name,
                "title": "Permalink to this headline",
            },
        )
        new_anchor.string = "¶"
        new_h3.append(new_anchor)
        dt.append(new_h3)
        hide_anchor = soup.new_tag(
            "a",
            attrs={
                "class": "hide-anchor",
                "name": method_name,
                "id": method_name,
            },
        )
        dt.insert(0, hide_anchor)
    with open(htmlfile, "w") as f:
        # f.write(soup.prettify())
        f.write(str(soup))


def filter_all_files(
    rootdir, ext="_en.html", action=insert_header_and_anchor_for_method
):
    """
    find all the _en.html file, and do the action.
    """
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            if f.endswith(ext):
                action(os.path.join(root, f))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="do sth after html files generated."
    )
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("dir", type=str, help="the file directory", default=".")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    filter_all_files(args.dir)
