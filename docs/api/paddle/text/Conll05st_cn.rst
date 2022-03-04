.. _cn_api_text_datasets_Conll05st:

Conll05st
-------------------------------

.. py:class:: paddle.text.datasets.Conll05st()


该类是对 `Conll05st <https://www.cs.upc.edu/~srlconll/soft.html>`_
测试数据集的实现.

.. note::
    只支持自动下载公共的 Conll05st测试数据集。

参数
:::::::::
    - data_file（str）- 保存数据的路径，如果参数 `download` 设置为True，可设置为None。默认为None。
    - word_dict_file（str）- 保存词典的路径。如果参数 `download` 设置为True，可设置为None。默认为None。
    - verb_dict_file（str）- 保存动词词典的路径。如果参数 `download` 设置为True，可设置为None。默认为None。
    - target_dict_file（str）- 保存目标词典的路径如果参数 `download` 设置为True，可设置为None。默认为None。
    - emb_file（str）- 保存词嵌入词典的文件。只有在 `get_embedding` 能被设置为None 且 `download` 为True时使用。
    - download（bool）- 如果 `data_file` 、 `word_dict_file` 、 `verb_dict_file` 和 `target_dict_file` 未设置，是否下载数据集。默认为True。

返回值
:::::::::
``Dataset``，conll05st数据集实例。

代码示例
:::::::::

.. code-block:: python

    import paddle
    from paddle.text.datasets import Conll05st

    class SimpleNet(paddle.nn.Layer):
        def __init__(self):
            super(SimpleNet, self).__init__()

        def forward(self, pred_idx, mark, label):
            return paddle.sum(pred_idx), paddle.sum(mark), paddle.sum(label)


    conll05st = Conll05st()

    for i in range(10):
        pred_idx, mark, label= conll05st[i][-3:]
        pred_idx = paddle.to_tensor(pred_idx)
        mark = paddle.to_tensor(mark)
        label = paddle.to_tensor(label)

        model = SimpleNet()
        pred_idx, mark, label= model(pred_idx, mark, label)
        print(pred_idx.numpy(), mark.numpy(), label.numpy())

