.. _cn_overview_text:

paddle.text
---------------------

paddle.text 目录是飞桨在文本领域的高层 API。有 Paddle 内置以及 PaddleNLP 中提供的两种。具体如下：

-  :ref:`内置数据集相关 API <about_datasets>`
-  :ref:`PaddleNLP 提供的 API <about_paddlenlp>`

.. _about_datasets:

内置数据集相关 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`Conll05st <cn_api_paddle_text_Conll05st>` ", "Conll05st 数据集"
    " :ref:`Imdb <cn_api_paddle_text_Imdb>` ", "Imdb 数据集"
    " :ref:`Imikolov <cn_api_paddle_text_Imikolov>` ", "Imikolov 数据集"
    " :ref:`Movielens <cn_api_paddle_text_Movielens>` ", "Movielens 数据集"
    " :ref:`UCIHousing <cn_api_paddle_text_UCIHousing>` ", "UCIHousing 数据集"
    " :ref:`WMT14 <cn_api_paddle_text_WMT14>` ", "WMT14 数据集"
    " :ref:`WMT16 <cn_api_paddle_text_WMT16>` ", "WMT16 数据集"

.. _about_paddlenlp:

PaddleNLP 提供的 API
::::::::::::::::::::

PaddleNLP 提供了在文本任务上简洁易用的全流程 API，旨在为飞桨开发者提升文本领域建模效率。深度适配飞桨框架，提供基于最新版 Paddle 的 NLP 领域最佳实践。

安装命令：

.. code-block::

    pip install --upgrade paddlenlp -i https://pypi.org/simple


可参考 PaddleNLP `GitHub <https://github.com/PaddlePaddle/PaddleNLP>`_ 以及 `文档 <https://paddlenlp.readthedocs.io/zh/latest/index.html>`_

.. csv-table::
    :header: "API 模块", "功能简介", "API 用法简单示例"
    :widths: 10, 20, 20

    " `paddlenlp.datasets <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html>`_ ", "数据集相关 API，包含自定义数据集，数据集贡献与数据集快速加载等功能", " ``train_ds = paddlenlp.datasets.load_dataset('ptb', splits='train')`` "
    " `paddlenlp.data <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/data_preprocess.html>`_ ", "文本数据处理 Pipeline 的相关 API", "见链接文档"
    " `paddlenlp.transformers <https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.html>`_ ", "基于 Transformer 结构相关的预训练模型 API，包含 ERNIE, BERT, RoBERTa, Electra 等主流经典结构和下游任务", " ``model = paddlenlp.transformers.BertForSequenceClassification.from_pretrained('bert-wwm-chinese', num_classes=2)`` "
    " `paddlenlp.metrics <https://paddlenlp.readthedocs.io/zh/latest/metrics/metrics.html>`_", "提供了文本任务上的一些模型评价指标，例如 Perplexity、GlLUE 中用到的评估器、BLEU、Rouge 等，与飞桨高层 API 兼容", " ``metric = paddlenlp.metrics.AccuracyAndF1()`` "
    " `paddlenlp.embeddings <https://paddlenlp.readthedocs.io/zh/latest/model_zoo/embeddings.html>`_", "词向量相关 API，支持一键快速加载包预训练的中文词向量，VisualDL 高维可视化等功能", " ``token_embedding = paddlenlp.embeddings.TokenEmbedding(embedding_name='fasttext.wiki-news.target.word-word.dim300.en')`` "
