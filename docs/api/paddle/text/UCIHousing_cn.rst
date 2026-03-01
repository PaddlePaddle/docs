.. _cn_api_paddle_text_UCIHousing:

UCIHousing
-------------------------------

.. py:class:: paddle.text.UCIHousing(data_file = None, mode = 'train', download = True)


该类是对 `UCI housing <https://github.com/rupakc/UCI-Data-Analysis/blob/master/Boston%20Housing%20Dataset/Boston%20Housing/UCI%20Machine%20Learning%20Repository_%20Housing%20Data%20Set.pdf>`_
测试数据集的实现。

参数
:::::::::

    - **data_file** (str) - 保存数据的路径，如果参数 :attr:`download` 设置为 True，可设置为 None。默认为 None。
    - **mode** (str) - 'train' 或 'test' 模式。默认为 'train'。
    - **download** (bool) - 如果 :attr:`data_file` 未设置，是否自动下载数据集。默认为 True。

返回值
:::::::::
``Dataset``，UCI housing 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.text.UCIHousing
