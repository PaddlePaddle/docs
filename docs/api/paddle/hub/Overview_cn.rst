.. _cn_overview_hub:

paddle.hub
-------------------------------

paddle.hub 是预训练模型库的集合，用来复用社区生产力，方便加载发布在 github、gitee 以及本地的预训练模型。飞桨提供框架模型拓展相关的 API 以及支持的模型库列表。具体如下：

-  :ref:`查看和加载 API <about_hub_functions>`
-  :ref:`支持模型库列表 <about_hub_repos>`


.. _about_hub_functions:

查看和加载 API
::::::::::::::::::::

.. csv-table::
    :header: "API 名称", "API 功能"
    :widths: 10, 30

    " :ref:`list <cn_api_paddle_hub_list>` ", "查看 Repo 支持的模型列表"
    " :ref:`help <cn_api_paddle_hub_help>` ", "查看指定模型的文档"
    " :ref:`load <cn_api_paddle_hub_load>` ", "加载指定模型"


.. _about_hub_repos:

支持模型列表
::::::::::::::::::::

.. csv-table::
    :header: "模型名字", "模型库"
    :widths: 10, 30

    "alexnet", "PaddleClas"
    "vgg11", "PaddleClas"
    "vgg13", "PaddleClas"
    "vgg16", "PaddleClas"
    "vgg19", "PaddleClas"
    "resnet18", "PaddleClas"
    "resnet34", "PaddleClas"
    "resnet50", "PaddleClas"
    "resnet101", "PaddleClas"
    "resnet152", "PaddleClas"
    "squeezenet1_0", "PaddleClas"
    "squeezenet1_1", "PaddleClas"
    "densenet121", "PaddleClas"
    "densenet161", "PaddleClas"
    "densenet169", "PaddleClas"
    "densenet201", "PaddleClas"
    "densenet264", "PaddleClas"
    "inceptionv3", "PaddleClas"
    "inceptionv4", "PaddleClas"
    "googlenet", "PaddleClas"
    "shufflenetv2_x0_25", "PaddleClas"
    "mobilenetv1", "PaddleClas"
    "mobilenetv1_x0_25", "PaddleClas"
    "mobilenetv1_x0_5", "PaddleClas"
    "mobilenetv1_x0_75", "PaddleClas"
    "mobilenetv2_x0_25", "PaddleClas"
    "mobilenetv2_x0_5", "PaddleClas"
    "mobilenetv2_x0_75", "PaddleClas"
    "mobilenetv2_x1_5", "PaddleClas"
    "mobilenetv2_x2_0", "PaddleClas"
    "mobilenetv3_large_x0_35", "PaddleClas"
    "mobilenetv3_large_x0_5", "PaddleClas"
    "mobilenetv3_large_x0_75", "PaddleClas"
    "mobilenetv3_large_x1_0", "PaddleClas"
    "mobilenetv3_large_x1_25", "PaddleClas"
    "mobilenetv3_small_x0_35", "PaddleClas"
    "mobilenetv3_small_x0_5", "PaddleClas"
    "mobilenetv3_small_x0_75", "PaddleClas"
    "mobilenetv3_small_x1_0", "PaddleClas"
    "mobilenetv3_small_x1_25", "PaddleClas"
    "resnext101_32x4d", "PaddleClas"
    "resnext101_64x4d", "PaddleClas"
    "resnext152_32x4d", "PaddleClas"
    "resnext152_64x4d", "PaddleClas"
    "resnext50_32x4d", "PaddleClas"
    "resnext50_64x4d", "PaddleClas"
    "bert", "PaddleNLP"



代码示例
:::::::::

.. code-block:: python

    import paddle

    # PaddleClas
    models = paddle.hub.list('PaddlePaddle/PaddleClas:develop', source='github', force_reload=True,)
    print(models)

    docs = paddle.hub.help('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=False,)
    print(docs)

    model = paddle.hub.load('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=False, pretrained=True)
    data = paddle.rand((1, 3, 224, 224))
    out = model(data)
    print(out.shape) # [1, 1000]


    # PaddleNLP
    docs = paddle.hub.help('PaddlePaddle/PaddleNLP:develop', model='bert',)
    print(docs)

    model, tokenizer = paddle.hub.load('PaddlePaddle/PaddleNLP:develop', model='bert', model_name_or_path='bert-base-cased')
