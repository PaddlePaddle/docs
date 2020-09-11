.. _cn_api_text_datasets_WMT14:

WMT14
-------------------------------

.. py:class:: paddle.text.datasets.WMT14()


    Implementation of `WMT14 <http://www.statmt.org/wmt14/>`_ test dataset.
    The original WMT14 dataset is too large and a small set of data for set is
    provided. This module will download dataset from
    http://paddlepaddle.bj.bcebos.com/demo/wmt_shrinked_data/wmt14.tgz

    参数
:::::::::
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'test' or 'gen'. Default 'train'
        dict_size(int): word dictionary size. Default -1.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of WMT14 dataset

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

            paddle.disable_static()

            wmt14 = WMT14(mode='train', dict_size=50)

            for i in range(10):
                src_ids, trg_ids, trg_ids_next = wmt14[i]
                src_ids = paddle.to_tensor(src_ids)
                trg_ids = paddle.to_tensor(trg_ids)
                trg_ids_next = paddle.to_tensor(trg_ids_next)

                model = SimpleNet()
                src_ids, trg_ids, trg_ids_next = model(src_ids, trg_ids, trg_ids_next)
                print(src_ids.numpy(), trg_ids.numpy(), trg_ids_next.numpy())

    