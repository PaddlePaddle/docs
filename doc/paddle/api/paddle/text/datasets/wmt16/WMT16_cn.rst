.. _cn_api_text_datasets_WMT16:

WMT16
-------------------------------

.. py:class:: paddle.text.datasets.WMT16()


该类是对`WMT16 <http://www.statmt.org/wmt16/>`_ 测试数据集实现。
ACL2016多模态机器翻译。有关更多详细信息，请访问此网站：
http://www.statmt.org/wmt16/multimodal-task.html#task1

如果您任务中使用了该数据集，请引用如下论文：
Multi30K: Multilingual English-German Image Descriptions.

.. code-block:: text

    @article{elliott-EtAl:2016:VL16,
        author    = {{Elliott}, D. and {Frank}, S. and {Sima"an}, K. and {Specia}, L.},
        title     = {Multi30K: Multilingual English-German Image Descriptions},
        booktitle = {Proceedings of the 6th Workshop on Vision and Language},
        year      = {2016},
        pages     = {70--74},
        year      = 2016
    }

参数
:::::::::
    - data_file（str）- 保存数据集压缩文件的路径，如果参数:attr:`download`设置为True，可设置为None。
    默认值为None。
    - mode（str）- 'train', 'test' 或 'val'。默认为'train'。
    - src_dict_size（int）- 源语言词典大小。默认为-1。
    - trg_dict_size（int) - 目标语言测点大小。默认为-1。
    - lang（str）- 源语言，'en' 或 'de'。默认为 'en'。
    - download（bool）- 如果:attr:`data_file`未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
``Dataset``，WMT16数据集实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.text.datasets import WMT16

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, src_ids, trg_ids, trg_ids_next):
            return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)

    paddle.disable_static()

    wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50)

    for i in range(10):
        src_ids, trg_ids, trg_ids_next = wmt16[i]
        src_ids = paddle.to_tensor(src_ids)
        trg_ids = paddle.to_tensor(trg_ids)
        trg_ids_next = paddle.to_tensor(trg_ids_next)

        model = SimpleNet()
        src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
        print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())

