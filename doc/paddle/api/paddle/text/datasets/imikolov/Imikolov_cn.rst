.. _cn_api_text_datasets_Imikolov:

Imikolov
-------------------------------

.. py:class:: paddle.text.datasets.Imikolov()


    Implementation of imikolov dataset.

    参数
:::::::::
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        data_type(str): 'NGRAM' or 'SEQ'. Default 'NGRAM'.
        window_size(int): sliding window size for 'NGRAM' data. Default -1.
        mode(str): 'train' 'test' mode. Default 'train'.
        min_word_freq(int): minimal word frequence for building word dictionary. Default 50.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of imikolov dataset

    代码示例
:::::::::

        .. code-block:: python

            import paddle
            from paddle.text.datasets import Imikolov

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super(SimpleNet, self).__init__()

                def forward(self, src, trg):
                    return paddle.sum(src), paddle.sum(trg)

            paddle.disable_static()

            imikolov = Imikolov(mode='train', data_type='SEQ', window_size=2)

            for i in range(10):
                src, trg = imikolov[i]
                src = paddle.to_tensor(src)
                trg = paddle.to_tensor(trg)

                model = SimpleNet()
                src, trg = model(src, trg)
                print(src.numpy().shape, trg.numpy().shape)

    