.. _cn_overview_callbacks:

paddle.audio
---------------------


paddle.audio 目录是飞桨在语音领域的高层 API。具体如下：

-  :ref:`音频特征相关 API <about_features>`
-  :ref:`音频处理基础函数相关 API <about_functional>`
-  :ref:`音频 I/O 相关 API <about_backends>`
-  :ref:`语音数据集相关 API <about_datasets>`

.. _about_features:

音频特征相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`LogMelSpectrogram <cn_api_paddle_audio_features_LogMelSpectrogram>` ", "计算语音特征 LogMelSpectrogram"
    " :ref:`MelSpectrogram <cn_api_paddle_audio_features_MelSpectrogram>` ", "计算语音特征 MelSpectrogram"
    " :ref:`MFCC <cn_api_paddle_audio_features_MFCC>` ", "计算语音特征 MFCC"
    " :ref:`Spectrogram <cn_api_paddle_audio_features_Spectrogram>` ", "计算语音特征 Spectrogram"

.. _about_functional:

音频处理基础函数相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`compute_fbank_matrix <cn_api_paddle_audio_functional_compute_fbank_matrix>` ", "计算 fbank 矩阵"
    " :ref:`create_dct <cn_api_paddle_audio_functional_create_dct>` ", "计算离散余弦变化矩阵"
    " :ref:`fft_frequencies <cn_api_paddle_audio_functional_fft_frequencies>` ", "计算离散傅里叶采样频率"
    " :ref:`hz_to_mel<cn_api_paddle_audio_functional_hz_to_mel>` ", "转换 hz 频率为 mel 频率"
    " :ref:`mel_to_hz<cn_api_paddle_audio_functional_mel_to_hz>` ", "转换 mel 频率为 hz 频率"
    " :ref:`mel_frequencies<cn_api_paddle_audio_functional_mel_frequencies>` ", "计算 mel 频率"
    " :ref:`power_to_db<cn_api_paddle_audio_functional_power_to_db>` ", "转换能量谱为分贝"
    " :ref:`get_window<cn_api_paddle_audio_functional_get_window>` ", "得到各种窗函数"

.. _about_backends:

音频 I/O 相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`get_current_backend <cn_api_paddle_audio_backends_get_current_backend>` ", "获取现在的语音 I/O 后端"
    " :ref:`list_available_backends <cn_api_paddle_audio_backends_list_available_backends>` ", "获取可设置得语音 I/O 后端"
    " :ref:`set_backend <cn_api_paddle_audio_backends_set_backend>` ", "设置语音 I/O 后端"
    " :ref:`load <cn_api_paddle_audio_load>` ", "载入音频"
    " :ref:`info <cn_api_paddle_audio_info>` ", "查询音频信息"
    " :ref:`save <cn_api_paddle_audio_save>` ", "保存音频"

.. _about_datasets:

音频数据集相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`TESS <cn_api_paddle_audio_datasets_TESS>` ", "TESS 数据集"
    " :ref:`ESC50 <cn_api_paddle_audio_datasets_ESC50>` ", "ESC50 数据集"
