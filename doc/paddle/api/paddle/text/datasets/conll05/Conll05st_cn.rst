.. _cn_api_text_datasets_Conll05st:

Conll05st
-------------------------------

.. py:class:: paddle.text.datasets.Conll05st()


    Implementation of `Conll05st <https://www.cs.upc.edu/~srlconll/soft.html>`_
    test dataset.

    Note: only support download test dataset automatically for that
          only test dataset of Conll05st is public.

    参数
:::::::::
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        word_dict_file(str): path to word dictionary file, can be set None if
            :attr:`download` is True. Default None
        verb_dict_file(str): path to verb dictionary file, can be set None if
            :attr:`download` is True. Default None
        target_dict_file(str): path to target dictionary file, can be set None if
            :attr:`download` is True. Default None
        emb_file(str): path to embedding dictionary file, only used for
            :code:`get_embedding` can be set None if :attr:`download` is
            True. Default None
        download(bool): whether to download dataset automatically if
            :attr:`data_file` :attr:`word_dict_file` :attr:`verb_dict_file`
            :attr:`target_dict_file` is not set. Default True

    Returns:
        Dataset: instance of conll05st dataset

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

            paddle.disable_static()

            conll05st = Conll05st()

            for i in range(10):
                pred_idx, mark, label= conll05st[i][-3:]
                pred_idx = paddle.to_tensor(pred_idx)
                mark = paddle.to_tensor(mark)
                label = paddle.to_tensor(label)

                model = SimpleNet()
                pred_idx, mark, label= model(pred_idx, mark, label)
                print(pred_idx.numpy(), mark.numpy(), label.numpy())

    