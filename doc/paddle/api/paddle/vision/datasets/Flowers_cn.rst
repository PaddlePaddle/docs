.. _cn_api_vision_datasets_Flowers:

Flowers
-------------------------------

.. py:class:: paddle.vision.datasets.Flowers()


    Implementation of `Flowers <https://www.robots.ox.ac.uk/~vgg/data/flowers/>`_
    dataset

    参数
:::::::::
        data_file(str): path to data file, can be set None if
            :attr:`download` is True. Default None
        label_file(str): path to label file, can be set None if
            :attr:`download` is True. Default None
        setid_file(str): path to subset index file, can be set
            None if :attr:`download` is True. Default None
        mode(str): 'train', 'valid' or 'test' mode. Default 'train'.
        transform(callable): transform to perform on image, None for on transform.
        download(bool): whether to download dataset automatically if
            :attr:`data_file` is not set. Default True

    代码示例
:::::::::
        
        .. code-block:: python

            from paddle.vision.datasets import Flowers

            flowers = Flowers(mode='test')

            for i in range(len(flowers)):
                sample = flowers[i]
                print(sample[0].shape, sample[1])

    