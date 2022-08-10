.. _cn_api_text_datasets_WMT16:

WMT16
-------------------------------

.. py:class:: paddle.text.datasets.WMT16()


该类是对 `WMT16 <http://www.statmt.org/wmt16/>`_ 测试数据集实现。
ACL2016 多模态机器翻译。有关更多详细信息，请访问此网站：
http://www.statmt.org/wmt16/multimodal-task.html#task1

如果您任务中使用了该数据集，请引用论文：`Multi30K: Multilingual English-German Image Descriptions. <https://aclanthology.org/W16-3210/>`_ 。

参数
:::::::::
    - **data_file**（str）- 保存数据集压缩文件的路径，如果参数 :attr:`download`设置为 True，可设置为 None。默认值为 None。
    - **mode**（str）- 'train'，'test' 或 'val'。默认为'train'。
    - **src_dict_size**（int）- 源语言词典大小。默认为-1。
    - **trg_dict_size**（int) - 目标语言测点大小。默认为-1。
    - **lang**（str）- 源语言，'en' 或 'de'。默认为 'en'。
    - **download**（bool）- 如果 :attr:`data_file`未设置，是否自动下载数据集。默认为 True。

返回值
:::::::::
``Dataset``，WMT16 数据集实例。实例一共有三个字段：

  - **src_ids** (np.array) - 源语言当前的 token id 序列。
  - **trg_ids** (np.array) - 目标语言当前的 token id 序列。
  - **trg_ids_next** (np.array) - 目标语言下一段的 token id 序列。

代码示例
:::::::::

COPY-FROM: paddle.text.datasets.WMT16
