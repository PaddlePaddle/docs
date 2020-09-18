# 框架类FAQ


## 数据处理

##### 问题：如何处理图片小但数量很大的数据集？

+ 答复：`multiprocess_reader`可以解决该问题，具体可参考[Github示例](https://github.com/PaddlePaddle/Paddle/issues/16592)。

----------

##### 问题：使用`py_reader`读取数据时，如何给变量命名？

+ 答复：可以通过设置里面的name变量。具体方法请参考飞桨[create_py_reader_by_data](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.3/api_cn/layers_cn.html#create-py-reader-by-data) API。

----------

##### 问题：使用多卡或多GPU进行数据并行时，如何设置异步数据读取？

+ 答复：使用多卡或多GPU进行数据并行时，需要设置：`places = fluid.cuda_places() if USE_CUDA else fluid.cpu_places(CPU_NUM)`，具体内容可以参考文档：[异步数据读取](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/prepare_data/use_py_reader.html) 。

----------

##### 问题：使用`paddle.dataset.mnist.train()`获得数据后，如何转换为可操作的Tensor？

+ 答复：执行`fluid.dygraph.to_varibale()`，将data数据转化为可以操作的动态图Tensor。

----------

##### 问题：如何给图片添加一个通道数，并进行训练？

+ 答复：执行`np.expand_dims`增加维度后再reshape。如果需要通道合并，可以执行`fluid.layers.concat()`。

----------

##### 问题：`paddle.fluid.layers.py_reader`和`fluid.io.PyReader`有什么区别？

+ 答复：两个都是异步的。推荐使用`fluid.io.PyReader`。

----------

##### 问题：有拓展Tensor维度的Op吗？

+ 答复：有的，操作方法请参考[unsqueeze op](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn/unsqueeze_cn.html) 。

----------

##### 问题：是否支持两维以上的变长tensor，如：shape[-1, -1, 128]？

+ 答复：配置网络时`shape`可以设置为：[-1，*任意整数*，128]，输入时`shape`可以设置为：[*任意整数，**任意整数*，128]。维度只是个占位，网络运行时的实际维度是从输入数据中推导出来的。两个"任意整数" 在输入和配置时可以不相等，但是配置网络时，第一维度必须为-1。

----------

##### 问题：如何从np.array生成一个具有Shape和DType的Tensor?

+ 答复：具体方法可参考文档 [LoD-Tensor使用说明]( https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/user_guides/howto/basic_concept/lod_tensor.html) 。

----------

##### 问题：如何初始化一个随机数的Tensor？

+ 答复：



  ta = fluid.create_lod_tensor(np.random.randn(10, 5), [], fluid.CPUPlace())

  tb = fluid.create_lod_tensor(np.ones([5, 10]), [], place)

  print(np.array(ta))

  print(np.array(tb))



## 模型搭建

##### 问题：如何不训练某层的权重？

+ 答复：在`ParamAttr`里设置learning_rate=0。

----------

##### 问题：`stop_gradient=True`的影响范围？

+ 答复：如果fluid里某一层使用`stop_gradient=True`，那么这一层之前的层都会自动 `stop_gradient=True`，梯度不再回传。

----------

##### 问题：请问`fluid.layers.matmul`和`fluid.layers.mul`有什么区别？

+ 答复：`matmul`支持broadcast和任意阶相乘。`mul`会把输入都转成两维去做矩阵乘。

----------



## 模型训练&评估

##### 问题：在CPU上进行模型训练，如何使用多线程？

+ 答复：可以参考使用[CompiledProgram API 中的with_data_parallel方法](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/CompiledProgram_cn.html#id5)。

----------


##### 问题：使用NVIDIA多卡运行Paddle时报错，`Error：NCCL ContextMap`或者`Error：hang住`（log日志打印突然卡住），如何解决？

+ 答复：参考[NCCL Tests](https://github.com/NVIDIA/nccl-tests)检测您的环境。如果检测不通过，请登录[NCCL官网](https://developer.nvidia.com/zh-cn)下载NCCl，安装后重新检测。

----------

##### 问题：多卡训练时启动失败，`Error：Out of all 4 Trainers`，如何处理？

+ 问题描述：多卡训练时启动失败，显示如下信息：

![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-13d1b5df218cb40b0243d13450ab667f34aee2f7)

+ 报错分析：主进程发现一号卡（逻辑）上的训练进程退出了。

+ 解决方法：查看一号卡上的日志，找出具体的出错原因。`paddle.distributed.launch` 启动多卡训练时，设置 `--log_dir` 参数会将每张卡的日志保存在设置的文件夹下。

----------

##### 问题：训练过程中提示显存不足，报错 `Error：Out of memory error GPU`，如何处理？

+ 答复：这是一种常见情况，可以先检查一下GPU卡是否被其他程序占用，你可以尝试调整`batch_size`大小，也可以更改网络模型，或者参考官方文档[存储分配与优化](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/singlenode_training_improving/memory_optimize.html) 。建议用户使用[AI Studio 学习与 实训社区训练](https://aistudio.baidu.com/aistudio/index)，获取免费GPU算力，显存16GB的v100，速度更快。

----------

##### 问题：如何提升模型训练时的GPU利用率？

+ 答复：有如下两点建议：

  1. 如果数据预处理耗时较长，可使用[fluid.io.DataLLoader](https://www.paddlepaddle.org.cn/documentation/docs/en/api/io/DataLoader.html#dataloader)加速；该API提供了多进程的异步加载支持，在静态图与动态图下均可使用，也是paddle后续主推的数据读取方式。用户可通过配置num_workers指定异步加载数据的进程数目从而满足不同规模数据集的读取需求。

  2. 如果GPU显存没满，且数据读取不是性能瓶颈，可以增加`batch_size`，但是注意调节其他超参数。

  以上两点均为比较通用的方案，其他的优化方案和模型相关，可参考相应models示例。

----------

##### 问题：使用CPU或GPU时，如何设置`num_threds`？

+ 答复：可以通过`os.getenv("CPU_NUM")`获取相关参数值或者`os.environ['CPU_NUM'] = str(2)`设置相关参数值。

----------

##### 问题：如何处理变长ID导致程序显存占用过大的问题？

+ 答复：请先参考上述显存不足的问题的解决方案。若存储空间仍然不够，建议对index进行排序，减少padding的数量。

----------

##### 问题：Executor与ParallelExecutor有什么区别？

+ 答复：飞桨（PaddlePaddle，以下简称Paddle）的设计思想类似于高级编程语言C++和JAVA等。程序的执行过程被分为编译和执行两个阶段。

用户完成对 Program 的定义后，Executor 接受这段 Program 并转化为C++后端真正可执行的 FluidProgram，这一自动完成的过程叫做编译。

编译过后需要 Executor 来执行这段编译好的 FluidProgram。

1. `fluid.Executor`执行对象是Program，可以认为是一个轻量级的执行器，目前主要用于参数初始化、参数加载、参数模型保存。

2. `fluid.ParallelExecutor` 是 `Executor` 的一个升级版本，可以支持基于数据并行的多节点模型训练和测试。如果采用数据并行模式， ParallelExecutor 在构造时会将参数分发到不同的节点上，并将输入的 Program 拷贝到不同的节点，在执行过程中，各个节点独立运行模型，将模型反向计算得到的参数梯度在多个节点之间进行聚合，之后各个节点独立的进行参数的更新。

----------

##### 问题：训练过程中如果出现不收敛的情况，如何处理？

+ 答复：不收敛的原因有很多，可以参考如下方式排查：

  1. 检查数据集中训练数据的准确率，数据是否有很多错误，特征是否归一化；
  2. 简化网络结构，先基于benchmark实验，确保在baseline网络结构和数据集上的收敛结果正确；
  3. 对于复杂的网络，每次只增加一个改动，确保改动后的网络正确；
  4. 检查网络在训练数据上的Loss是否下降；
  5. 检查学习率、优化算法是否合适，学习率过大会导致不收敛；
  6. 检查`batch_size`设置是否合适，`batch_size`过小会导致不收敛；
  7. 检查梯度计算是否正确，是否有梯度过大的情况，是否为NaN。

----------

##### 问题：Loss为NaN，如何处理？

+ 答复：可能由于网络的设计问题，Loss过大（Loss为NaN）会导致梯度爆炸。如果没有改网络结构，但是出现了NaN，可能是数据读取导致，比如标签对应关系错误。还可以检查下网络中是否会出现除0，log0的操作等。

----------

##### 问题：在AI Studio上使用GPU训练时报错 `Attempt to use GPU for prediction, but environment variable CUDA_VISIBLE_DEVICES was not set correctly.`，如何处理？

+ 答复：需要在Notebook环境中增加：`%set_env CUDA_VISIBLE_DEVICES=0`。

----------

##### 问题：使用GPU训练时报错，`Error：incompatible constructor arguments.`，如何处理？

+ 问题描述：
  ![图片](https://agroup-bos-bj.cdn.bcebos.com/bj-3779aa5b33dbe1f05ba2bfeabb2d22d4270d1929)

+ 报错分析：`CUDAPlace()`接口没有指定GPU的ID编号导致。

+ 答复：CUDAPlace()接口需要指定GPU的ID编号，接口使用方法参见：[paddle.fluid.CUDAPlace](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/CUDAPlace_cn.html)。

----------

##### 问题：出现未编译CUDA报错怎么办？

![](https://ai-studio-static-online.cdn.bcebos.com/aba33440dd194ea397528f06bcb3574bddcf496b679b4da2832955b71cf65c76)

* 答复：报错是由于没有安装GPU版本的PaddlePaddle，CPU版本默认不包含CUDA检测功能。使用`pip install paddlepaddle-gpu -U` 即可。

-----

##### 问题：增量训练中，如何保存模型和恢复训练？

+ 答复：在增量训练过程中，不仅需要保存模型的参数，也需要保存模型训练的状态（如learning_rate）。使用[save](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/save_cn.html#save)保存模型训练的参数和状态；恢复训练时，使用[load](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/fluid_cn/load_cn.html#load)进行恢复训练。

----------

##### 问题：训练后的模型很大，如何压缩？

+ 答复：建议您使用飞桨模型压缩工具[PaddleSlim](https://www.paddlepaddle.org.cn/tutorials/projectdetail/489539)。PaddleSlim是飞桨开源的模型压缩工具库，包含模型剪裁、定点量化、知识蒸馏、超参搜索和模型结构搜索等一系列模型压缩策略，专注于**模型小型化技术**。

----------



## 参数调整

##### 问题：如何将本地数据传入`fluid.dygraph.Embedding`的参数矩阵中？

+ 答复：需将本地词典向量读取为NumPy数据格式，然后使用`fluid.initializer.NumpyArrayInitializer`这个op初始化`fluid.dygraph.Embedding`里的`param_attr`参数，即可实现加载用户自定义（或预训练）的Embedding向量。

------

##### 问题：如何实现网络层中多个feature间共享该层的向量权重？

+ 答复：将所有网络层中`param_attr`参数里的`name`设置为同一个，即可实现共享向量权重。如使用embedding层时，可以设置`param_attr=fluid.ParamAttr(name="word_embedding")`，然后把param_attr传入embedding中。

----------

##### 问题：如何修改全连接层参数，如：weights、bias、optimizer.SGD？

+ 答复：可以通过`param_attr`设置参数的属性，`fluid.ParamAttr( initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)`，如果`learning_rate`设置为0，该层就不参与训练。手动输入参数也可以实现，但是会比较麻烦。

----------

##### 问题：使用optimizer或ParamAttr设置的正则化和学习率，二者什么差异？

+ 答复：ParamAttr中定义的`regularizer`优先级更高。若ParamAttr中定义了`regularizer`，则忽略Optimizer中的`regularizer`；否则，则使用Optimizer中的`regularizer`。ParamAttr中的学习率默认为1，在对参数优化时，最终的学习率等于optimizer的学习率乘以ParamAttr的学习率。

----------

##### 问题：如何导出指定层的权重，如导出最后一层的*weights*和*bias*？

+ 答复：使用`save_vars`保存指定的vars，然后使用`load_vars`加载对应层的参数值。具体示例请见API文档：[load_vars](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/load_vars_cn.html#load-vars) 和 [save_vars](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/io_cn/save_vars_cn.html#save-vars) 。

----------

##### 问题：训练过程中如何固定网络和Batch Normalization（BN）？

+ 答复：

1. 对于固定BN：设置 `use_global_stats=True`，使用已加载的全局均值和方差：`global mean/variance`，具体内容可查看官网文档[BatchNorm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/dygraph_cn/BatchNorm_cn.html)。

2. 对于固定网络层：如： stage1→ stage2 → stage3 ，设置stage2的输出，假设为*y*，设置 `y.stop_gradient=True`，那么， stage1→ stage2整体都固定了，不再更新。

----------

##### 问题：优化器设置时报错`AttributeError: parameter_list argument given to the Optimizer should not be None in dygraph mode.`，如何处理？

+ 错误分析：必选参数缺失导致。

+ 答复：飞桨框架1.7版本之后，动态图模式下，需要在optimizer的设置中加入必选项`parameter_list`。

----------

##### 问题：`fluid.layer.pool2d`的全局池化参数和设置参数有关系么？

+ 答复：如果设置`global_pooling`，则设置的`pool_size`将忽略，不会产生影响。

----------

##### 问题：训练的step在参数优化器中是如何变化的？

<img src="https://ai-studio-static-online.cdn.bcebos.com/610cd445435e40e1b1d8a4944a7448c35d89ea33ab364ad8b6804b8dd947e88c" style="zoom: 50%;" />

* 答复：

  `step`表示的是经历了多少组mini_batch，其统计方法为`exe.run`(对应Program)运行的当前次数，即每运行一次`exe.run`，step加1。举例代码如下：

```python
# 执行下方代码后相当于step增加了N x Epoch总数
for epoch in range(epochs):
	# 执行下方代码后step相当于自增了N
	for data in [mini_batch_1,2,3...N]:
		# 执行下方代码后step += 1
		exe.run(data)
```

-----


##### 问题：如何修改全连接层参数，比如weight，bias？

+ 答复：可以通过`param_attr`设置参数的属性，`fluid.ParamAttr( initializer=fluid.initializer.Normal(0.0, 0.02), learning_rate=2.0)`，如果`learning_rate`设置为0，该层就不参与训练。也可以构造一个numpy数据，使用`fluid.initializer.NumpyArrayInitializer`来给权重设置想要的值。

----------

## 应用预测

##### 问题：load_inference_model在加载预测模型时能否用py_reader读取？

+ 答复：目前`load_inference_model`加载进行的模型还不支持py_reader输入。

----------