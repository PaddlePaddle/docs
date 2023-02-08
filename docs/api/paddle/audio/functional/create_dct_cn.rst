.. _cn_api_audio_functional_create_dct:

create_dct
-------------------------------

.. py:function:: paddle.audio.functional.create_dct(n_mfcc, n_mels, norm='ortho', dtype='float32')

计算离散余弦变换矩阵。

参数
::::::::::::

    - **n_mfcc** (float) - mel 倒谱系数数目。
    - **n_mels** (int) - mel 的 fliterbank 数。
    - **norm** (float，可选) - 正则化类型，默认值是'ortho'。
    - **dtype** (str，可选) - 默认'float32'。

返回
:::::::::

``paddle.Tensor``，Tensor 形状 (n_mels, n_mfcc)。

代码示例
:::::::::

COPY-FROM: paddle.audio.functional.create_dct
