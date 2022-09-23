.. _cn_overview_callbacks:

paddle.audio
---------------------

paddle.audio 目录是飞桨在语音领域的高层 API。具体如下：

-  :ref:`音频特征相关 API <about_features>`
-  :ref:`音频处理基础函数相关 API <about_functional>`

.. _about_features:

音频特征相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`LogMelSpectrogram <cn_api_paddle_audio_features_LogMelSpectrogram>` ", "计算语音特征LogMelSpectrogram" 
    " :ref:`MelSpectrogram <cn_api_paddle_audio_features_MelSpectrogram>` ", "计算语音特征MelSpectrogram"
    " :ref:`MFCC <cn_api_audio_features_MFCC>` ", "计算语音特征MFCC"
    " :ref:`Spectrogram <cn_api_audio_features_Spectrogram>` ", "计算语音特征Spectrogram"

.. _about_functional:

音频处理基础函数相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`compute_fbank_matrix <cn_api_audio_functional_compute_fbank_matrix>` ", "计算fbank矩阵"
    " :ref:`create_dct <cn_api_audio_functional_create_dct>` ", "计算离散余弦变化矩阵"
    " :ref:`fft_frequencies <cn_api_audio_functional_fft_frequencies>` ", "计算离散傅里叶采样频率"
    " :ref:`hz_to_mel<cn_api_audio_functional_hz_to_mel>` ", "转换hz频率为mel频率"
    " :ref:`mel_to_hz<cn_api_audio_functional_mel_to_hz>` ", "转换mel频率为hz频率"
    " :ref:`mel_frequencies<cn_api_audio_functional_mel_frequencies>` ", "计算mel频率"
    " :ref:`power_to_db<cn_api_audio_functional_power_to_db>` ", "转换能量谱为分贝"
    " :ref:`get_window<cn_api_audio_functional_get_window>` ", "得到各种窗函数"

