.. _cn_api_paddle_text_WMT14:

WMT14
-------------------------------

.. py:class:: paddle.text.WMT14(data_file = None, mode = 'train', dict_size = -1, download = True)


该类是对 `WMT14 <http://www.statmt.org/wmt14/>`_ 测试数据集实现。
由于原始 WMT14 数据集太大，我们在这里提供了一组小数据集。该类将从
http://paddlemodels.bj.bcebos.com/wmt/wmt14.tgz
下载数据集。

参数
:::::::::
    - **data_file**（str）- 保存数据集压缩文件的路径，如果参数：attr: `download` 设置为 True，可设置为 None。默认为 None。

    - **mode**（str）- 'train'，'test' 或'gen'。默认为'train'。

    - **dict_size**（int）- 词典大小。默认为-1。

    - **download**（bool）- 如果：attr: `data_file` 未设置，是否自动下载数据集。默认为 True。

返回值
:::::::::
``Dataset``，WMT14 数据集实例。

  - **src_ids** (np.array) - 源语言当前的 token id 序列。
  - **trg_ids** (np.array) - 目标语言当前的 token id 序列。
  - **trg_ids_next** (np.array) - 目标语言下一段的 token id 序列。

代码示例
:::::::::

COPY-FROM: paddle.text.WMT14
