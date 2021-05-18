.. _cn_overview_text:

paddle.text
---------------------

paddle.text 目录是飞桨在文本领域的高层API。有Paddle内置以及PaddleNLP中提供的两种。具体如下：

-  :ref:`内置数据集相关API <about_datasets>`

.. _about_datasets:

内置数据集相关API
::::::::::::::::::::

.. csv-table::
    :header: "API名称", "API功能"
    :widths: 10, 30

    " :ref:`Conll05st <cn_api_text_datasets_Conll05st>` ", "Conll05st数据集"
    " :ref:`Imdb <cn_api_text_datasets_Imdb>` ", "Imdb数据集"
    " :ref:`Imikolov <cn_api_text_datasets_Imikolov>` ", "Imikolov数据集"
    " :ref:`Movielens <cn_api_text_datasets_Movielens>` ", "Movielens数据集"
    " :ref:`UCIHousing <cn_api_text_datasets_UCIHousing>` ", "UCIHousing数据集"
    " :ref:`WMT14 <cn_api_text_datasets_WMT14>` ", "WMT14数据集"
    " :ref:`WMT16 <cn_api_text_datasets_WMT16>` ", "WMT16数据集"

PaddleNLP提供的API
::::::::::::::::::::

PaddleNLP 2.0 提供了在文本任务上简洁易用的全流程API与动静统一的高性能分布式训练能力，旨在为飞桨开发者提升文本领域建模效率，并提供基于PaddlePaddle 2.0的NLP领域最佳实践。

安装命令：

.. code-block::

    pip install paddlenlp2.0.0


可参考项目 `GitHub <https://github.com/PaddlePaddle/PaddleNLP>`_ 以及 `文档 <https://paddlenlp.readthedocs.io/zh/latest/index.html>`_ 。

.. csv-table::
    :header: "API模块", "功能简介", "API用法简单示例"
    :widths: 20, 40，40

    " `paddlenlp.datasets <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html>`_ ", "提供文本分类、阅读理解、序列标注、机器翻译、文本生成等任务的多个数据集", " ``train_ds, dev_ds = paddlenlp.datasets.load_dataset('ptb', splits=('train', 'dev'))`` "
    " `paddlenlp.data <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/data_preprocess.html`_ ", "该模块提供了在NLP任务中构建有效的数据处理Pipeline的常用API", "见链接文档"
    " `paddlenlp.transformers <https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html>`_ ", "该模块提供了目前PaddleNLP支持的各类预训练模型及各项任务，例如Transformer, BERT, ERNIE, GPT, RoBERTa, XLNET等", " ``model = paddlenlp.transformers.BertForSequenceClassification.from_pretrained("bert-wwm-chinese", num_classes=2)`` "
    " `paddlenlp.metrics <https://paddlenlp.readthedocs.io/zh/latest/metrics/metrics.html>`_", "该模块提供了文本任务上的一些模型评价指标，例如Perplexity、GlLUE中用到的评估器、BLEU、Rouge等，与飞桨高层API兼容", " ``metric = paddlenlp.metrics.AccuracyAndF1()`` "
    " `paddlenlp.embeddings <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/embeddings.md>`_", "提供多个开源的预训练词向量模型", " ``token_embedding = paddlenlp.embeddings.TokenEmbedding(embedding_name="fasttext.wiki-news.target.word-word.dim300.en")`` "
