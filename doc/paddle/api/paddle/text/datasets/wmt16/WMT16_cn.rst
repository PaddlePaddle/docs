.. _cn_api_text_datasets_WMT16:

WMT16
-------------------------------

.. py:class:: paddle.text.datasets.WMT16()


    Implementation of `WMT16 <http://www.statmt.org/wmt16/>`_ test dataset.
    ACL2016 Multimodal Machine Translation. Please see this website for more
    details: http://www.statmt.org/wmt16/multimodal-task.html#task1

    If you use the dataset created for your task, please cite the following paper:
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
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train', 'test' or 'val'. Default 'train'
        src_dict_size(int): word dictionary size for source language word. Default -1.
        trg_dict_size(int): word dictionary size for target language word. Default -1.
        lang(str): source language, 'en' or 'de'. Default 'en'.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    Returns:
        Dataset: instance of WMT16 dataset

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

    