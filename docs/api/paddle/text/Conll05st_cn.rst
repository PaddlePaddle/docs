.. _cn_api_paddle_text_Conll05st:

Conll05st
-------------------------------

.. py:class:: paddle.text.datasets.Conll05st()


该类是对 `Conll05st <https://www.cs.upc.edu/~srlconll/soft.html>`_
测试数据集的实现。

.. note::
    只支持自动下载公共的 Conll05st 测试数据集。

参数
:::::::::
    - **data_file** (str) - 保存数据的路径，如果参数 `download` 设置为 True，可设置为 None。默认为 None。
    - **word_dict_file** (str) - 保存词典的路径。如果参数 `download` 设置为 True，可设置为 None。默认为 None。
    - **verb_dict_file** (str) - 保存动词词典的路径。如果参数 `download` 设置为 True，可设置为 None。默认为 None。
    - **target_dict_file** (str) - 保存目标词典的路径如果参数 `download` 设置为 True，可设置为 None。默认为 None。
    - **emb_file** (str) - 保存词嵌入词典的文件。只有在 `get_embedding` 能被设置为 None 且 `download` 为 True 时使用。
    - **download** (bool) - 如果 `data_file` 、 `word_dict_file` 、 `verb_dict_file` 和 `target_dict_file` 未设置，是否下载数据集。默认为 True。

返回值
:::::::::
``Dataset``，conll05st 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.text.Conll05st:code-example1
