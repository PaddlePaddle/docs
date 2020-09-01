模型可视化
==============

通过 `Quick Start <../introduction/quick_start.html>`_ 一节中，我们了解到，预测模型包含了两个文件，一部分为模型结构文件，通常以 **model** 或 **__model__** 文件存在；另一部分为参数文件，通常以params 文件或一堆分散的文件存在。

模型结构文件，顾名思义，存储了模型的拓扑结构，其中包括模型中各种OP的计算顺序以及OP的详细信息。很多时候，我们希望能够将这些模型的结构以及内部信息可视化，方便我们进行模型分析。接下来将会通过两种方式来讲述如何对Paddle 预测模型进行可视化。

一： 通过 VisualDL 可视化
------------------

1） 安装

VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等，帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，实现高效的模型优化。
我们可以进入 `GitHub主页 <https://github.com/PaddlePaddle/VisualDL#%E5%AE%89%E8%A3%85%E6%96%B9%E5%BC%8F>`_ 进行下载安装。

2）可视化

`点击 <https://paddle-inference-dist.bj.bcebos.com/temp_data/sample_model/__model__>`_ 下载测试模型。

支持两种启动方式：

- 前端拖拽上传模型文件：

  - 无需添加任何参数，在命令行执行 visualdl 后启动界面上传文件即可：


.. image:: https://user-images.githubusercontent.com/48054808/88628504-a8b66980-d0e0-11ea-908b-196d02ed1fa2.png


- 后端透传模型文件：

  - 在命令行加入参数 --model 并指定 **模型文件** 路径（非文件夹路径），即可启动：

.. code:: python

  visualdl --model ./log/model --port 8080


.. image:: https://user-images.githubusercontent.com/48054808/88621327-b664f280-d0d2-11ea-9e76-e3fcfeea4e57.png

Graph功能详细使用，请见 `Graph使用指南 <https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README.md#Graph--%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E7%BB%84%E4%BB%B6>`_ 。

二： 通过代码方式生成dot文件
---------------------

1）pip 安装Paddle

2）生成dot文件

`点击 <https://paddle-inference-dist.bj.bcebos.com/temp_data/sample_model/__model__>`_ 下载测试模型。

.. code:: python

	#!/usr/bin/env python
	import paddle.fluid as fluid
	from paddle.fluid import core
	from paddle.fluid.framework import IrGraph

	def get_graph(program_path):
	    with open(program_path, 'rb') as f:
		    binary_str = f.read()
	    program =   fluid.framework.Program.parse_from_string(binary_str)
	    return IrGraph(core.Graph(program.desc), for_test=True)

	if __name__ == '__main__':
	    program_path = './lecture_model/__model__' 
	    offline_graph = get_graph(program_path)
	    offline_graph.draw('.', 'test_model', [])


3）生成svg

**Note：需要环境中安装graphviz**

.. code:: python

	dot -Tsvg ./test_mode.dot -o test_model.svg
	

然后将test_model.svg以浏览器打开预览即可。

.. image::  https://user-images.githubusercontent.com/5595332/81796500-19b59e80-9540-11ea-8c70-31122e969683.png
