.. _cn_api_vision_datasets_Flowers:

Flowers
-------------------------------

.. py:class:: paddle.vision.datasets.Flowers()


    `Flowers <https://www.robots.ox.ac.uk/~vgg/data/flowers/>`_ 数据集

参数
:::::::::
        - data_file (str) - 数据文件路径，如果 ``download`` 参数设置为 ``True`` ， ``data_file`` 参数可以设置为 ``None`` 。默认值为 ``None`` 。
        - label_file (str) - 标签文件路径，如果 ``download`` 参数设置为 ``True`` ， ``label_file`` 参数可以设置为 ``None`` 。默认值为 ``None`` 。
        - setid_file (str) - 子数据集下标划分文件路径，如果 ``download`` 参数设置为 ``True`` ， ``setid_file`` 参数可以设置为 ``None`` 。默认值为 ``None`` 。
        - mode (str) - ``'train'`` 或 ``'test'`` 模式，默认为 ``'train'`` 。
        - transform (callable) - 图片数据的预处理，若为 ``None`` 即为不做预处理。默认值为 ``None``。
        - download (bool) - 当 ``data_file`` 是 ``None`` 时，该参数决定是否自动下载数据集文件。默认为 ``True`` 。

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
                print(sample[0].size, sample[1])

    
