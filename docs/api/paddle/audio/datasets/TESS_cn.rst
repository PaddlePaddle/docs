.. _cn_api_audio_datasets_TESS:

TESS
-------------------------------

.. py:class:: paddle.audio.datasets.TESS(mode: str = 'train', seed = 0, n_folds = 5, split = 1, feat_type = 'raw', archive=None, **kwargs)


`TESS <https://tspace.library.utoronto.ca/handle/1807/24487>`_ 数据集的实现。

参数
:::::::::

  - **mode** (str，可选) - ``'train'`` 或 ``'dev'`` 模式两者之一，默认值为 ``'train'``。
  - **seed** (int) - 默认是0，指定随机数来对数据进行重新排序。
  - **n_folds** (int) - 默认是5，指定把数据集分为的文件夹数目， 1个文件夹是dev，其他是train。
  - **split** (int) - 默认是1，指定dev的文件夹。
  - **feat_type** (str) - 默认是raw，raw是原始语音，支持mfcc，spectrogram，melspectrogram，logmelspectrogram。指定从音频提取的语音特征。
  - **archive** (dict) - 默认是None，类中已经设置默认archive，指定数据集的下载链接和md5值。

返回
:::::::::

:ref:`cn_api_io_cn_Dataset`，TESS 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.audio.datasets.TESS
