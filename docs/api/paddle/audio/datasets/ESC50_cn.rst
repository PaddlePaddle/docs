.. _cn_api_audio_datasets_ESC50:

ESC50
-------------------------------

.. py:class:: paddle.audio.datasets.ESC50(mode: str = 'train', split: int = 1, feat_type: str = 'raw', archive=None, **kwargs)


`ESC50 <http://dx.doi.org/10.1145/2733373.2806390>`_ 数据集的实现。

参数
:::::::::

  - **mode** (str，可选) - ``'train'`` 或 ``'dev'`` 模式两者之一，默认值为 ``'train'``。
  - **split** (int) - 默认是1，指定dev的文件夹。
  - **feat_type** (str) - 默认是raw，raw是原始语音，支持mfcc，spectrogram，melspectrogram，logmelspectrogram。指定从音频提取的语音特征。
  - **archive** (dict) - 默认是None，类中已经设置默认archive，指定数据集的下载链接和md5值。

返回
:::::::::

:ref:`cn_api_io_cn_Dataset`，ESC50 数据集实例。

代码示例
:::::::::

COPY-FROM: paddle.audio.datasets.ESC50
