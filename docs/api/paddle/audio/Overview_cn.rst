.. _cn_overview_callbacks:

paddle.audio
---------------------

paddle.audio 目录是飞桨在语音领域的高层 API。具体如下：

-  :ref:`音频 I/O 相关 API <about_backends>`
-  :ref:`语音数据集相关 API <about_datasets>`

.. _about_backends:

音频特征相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`get_current_audio_backend <cn_api_audio_backends_get_current_audio_backend>` ", "获取现在的语音 I/O 后端"
    " :ref:`list_available_backends <cn_api_audio_backends_list_available_backends>` ", "获取可设置得语音 I/O 后端"
    " :ref:`set_backend <cn_api_audio_backends_set_backend>` ", "设置语音 I/O 后端"
    " :ref:`load <cn_api_audio_backends_load>` ", "载入音频"
    " :ref:`info <cn_api_audio_backends_info>` ", "查询音频信息"
    " :ref:`save <cn_api_audio_backends_save>` ", "保存音频"

.. _about_datasets:

音频数据集相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`TESS <cn_api_audio_datasets_TESS>` ", "TESS 数据集"
    " :ref:`ESC50 <cn_api_audio_datasets_ESC50>` ", "ESC50 数据集"
