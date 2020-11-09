.. _cn_api_vision_datasets_Flowers:

Flowers
-------------------------------

.. py:class:: paddle.vision.datasets.Flowers()


    `Flowers <https://www.robots.ox.ac.uk/~vgg/data/flowers/>`_ 数据集

参数
:::::::::
        - data_file (str) - 数据文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - label_file (str) - 标签文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - setid_file (str) - 子数据集下标划分文件路径，如果 ``download`` 设置为 ``True`` ，此参数可以设置为None。默认值为None。
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - transform (callable) - 作用于图片数据的transform，若未 ``None`` 即为无transform。
        - download (bool) - 是否自定下载数据集文件。默认为 ``True`` 。

返回
:::::::::

				Flowers数据集实例

代码示例
:::::::::
        
        .. code-block:: python

            from paddle.vision.datasets import Flowers

            flowers = Flowers(mode='test')

            for i in range(len(flowers)):
                sample = flowers[i]
                print(sample[0].shape, sample[1])

    
