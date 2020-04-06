.. _cn_api_fluid_io_Flowers:

Flowers
-------------------------------

.. py:class:: paddle.fluid.io.Flowers(data_file=None, label_file=None, setid_file=None, mode='train', download=True)

Flowers数据集

参数:
    - **data_file** (str) - 数据集数据文件路径，若 ``download`` 为True， ``data_file`` 可设置为None。默认值为None。
    - **label_file** (str) - 数据集数据文件路径，若 ``download`` 为True， ``label_file`` 可设置为None。默认值为None。
    - **setid_file** (str) - 数据集数据文件路径，若 ``download`` 为True， ``setid_file`` 可设置为None。默认值为None。
    - **mode** (str) - 数据集模式，即读取 ``'train'`` ``valid`` 或者 ``'test'`` 数据。默认值为 ``'train'`` 。
    - **download** (bool) - 当 ``data_file`` ``label_file`` 或 ``setid_file`` 为None时，是否自动下载数据集。默认值为True。

返回：Flowers数据集

返回类型: Dataset

**代码示例**

.. code-block:: python

    from paddle.fluid.io import Flowers

    flowers = Flowers(mode='test')

    for i in range(len(flowers)):
        sample = flowers[i]
        print(sample[0].shape, sample[1])
