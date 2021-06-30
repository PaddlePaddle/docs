.. _cn_api_text_datasets_WMT14:

WMT14
-------------------------------

.. py:class:: paddle.text.datasets.WMT14()


该类是对 `WMT14 <http://www.statmt.org/wmt14/>`_ 测试数据集实现。
由于原始WMT14数据集太大，我们在这里提供了一组小数据集。该类将从
http://paddlemodels.bj.bcebos.com/wmt/wmt14.tgz
下载数据集。

参数
:::::::::
    - data_file（str）- 保存数据集压缩文件的路径, 如果参数:attr: `download` 设置为True，可设置为None。默认为None。

    - mode（str）- 'train', 'test' 或'gen'。默认为'train'。

    - dict_size（int）- 词典大小。默认为-1。

    - download（bool）- 如果:attr: `data_file` 未设置，是否自动下载数据集。默认为True。

返回值
:::::::::
``Dataset``，WMT14数据集实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.text.datasets import WMT14

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, src_ids, trg_ids, trg_ids_next):
            return paddle.sum(src_ids), paddle.sum(trg_ids), paddle.sum(trg_ids_next)


    wmt14 = WMT14(mode='train', dict_size=50)

    for i in range(10):
        src_ids, trg_ids, trg_ids_next = wmt14[i]
        src_ids = paddle.to_tensor(src_ids)
        trg_ids = paddle.to_tensor(trg_ids)
        trg_ids_next = paddle.to_tensor(trg_ids_next)

        model = SimpleNet()
        src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
        print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())
