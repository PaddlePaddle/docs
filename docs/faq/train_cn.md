# 组网、训练、评估常见问题

##### 问题：`stop_gradient=True`的影响范围？

+ 答复：如果静态图中某一层使用`stop_gradient=True`，那么这一层之前的层都会自动 `stop_gradient=True`，梯度不再回传。

----------

##### 问题：请问`paddle.matmul`和`paddle.multiply`有什么区别？

+ 答复：`matmul`支持的两个tensor的矩阵乘操作。`muliply`是支持两个tensor进行逐元素相乘。

----------

##### 问题：在模型组网时，inplace参数的设置会影响梯度回传吗？经过不带参数的op之后，梯度是否会保留下来？

+ 答复：inplace 参数不会影响梯度回传。只要用户没有手动设置`stop_gradient=True`，梯度都会保留下来。

----------

##### 问题：如何不训练某层的权重？

+ 答复：在`ParamAttr`里设置`learning_rate=0`或`trainable`设置为`False`。具体请[参考文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/fluid/param_attr/ParamAttr_cn.html)

----------

##### 问题：使用CPU进行模型训练，如何利用多处理器进行加速？

+ 答复：在2.0版本动态图模式下，CPU训练加速可以从以下两点进行配置：

1. 使用多进程DataLoader加速数据读取：训练数据较多时，数据处理往往会成为训练速度的瓶颈，paddle提供了异步数据读取接口DataLoader，可以使用多进程进行数据加载，充分利用多处理的优势，具体使用方法及示例请参考API文档：[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)。

2. 推荐使用支持[MKL（英特尔数学核心函数库）](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html)的paddle安装包，MKL相比Openblas等通用计算库在计算速度上有显著的优势，能够提升您的训练效率。

----------

##### 问题：使用NVIDIA多卡运行Paddle时报错 Nccl error，如何解决？

+ 答复：这个错误大概率是环境配置不正确导致的，建议您使用NVIDIA官方提供的方法参考检测自己的环境是否配置正确。具体地，可以使用[ NCCL Tests ](https://github.com/NVIDIA/nccl-tests) 检测您的环境；如果检测不通过，请登录[ NCCL官网 ](https://developer.nvidia.com/zh-cn)下载NCCl，安装后重新检测。

----------

##### 问题：多卡训练时启动失败，`Error：Out of all 4 Trainers`，如何处理？

+ 问题描述：多卡训练时启动失败，显示如下信息：

![图片](https://paddlepaddleimage.cdn.bcebos.com/faqimage%2Fbj-13d1b5df218cb40b0243d13450ab667f34aee2f7.png)

+ 报错分析：主进程发现一号卡（逻辑）上的训练进程退出了。

+ 解决方法：查看一号卡上的日志，找出具体的出错原因。`paddle.distributed.launch` 启动多卡训练时，设置 `--log_dir` 参数会将每张卡的日志保存在设置的文件夹下。

----------

##### 问题：训练时报错提示显存不足，如何解决？

+ 答复：可以尝试按如下方法解决：

1. 检查是当前模型是否占用了过多显存，可尝试减小`batch_size` ；

2. 开启以下三个选项：

```bash
#一旦不再使用即释放内存垃圾，=1.0 垃圾占用内存大小达到10G时，释放内存垃圾
export FLAGS_eager_delete_tensor_gb=0.0
#启用快速垃圾回收策略，不等待cuda kernel 结束，直接释放显存
export FLAGS_fast_eager_deletion_mode=1
#该环境变量设置只占用0%的显存
export FLAGS_fraction_of_gpu_memory_to_use=0
```

详细请参考官方文档[存储分配与优化](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/singlenode_training_improving/memory_optimize.html) 调整相关配置。

此外，建议您使用[AI Studio 学习与 实训社区训练](https://aistudio.baidu.com/aistudio/index)，获取免费GPU算力，提升您的训练效率。

----------

##### 问题：如何提升模型训练时的GPU利用率？

+ 答复：有如下两点建议：

  1. 如果数据预处理耗时较长，可使用DataLoader加速数据读取过程，具体请参考API文档：[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DataLoader_cn.html#dataloader)。

  2. 如果提高GPU计算量，可以增大`batch_size`，但是注意同时调节其他超参数以确保训练配置的正确性。

  以上两点均为比较通用的方案，其他的优化方案和模型相关，可参考官方模型库 [models](https://github.com/PaddlePaddle/models) 中的具体示例。

----------

##### 问题：如何处理变长ID导致程序内存占用过大的问题？

+ 答复：请先参考[显存分配与优化文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/performance_improving/singlenode_training_improving/memory_optimize.html) 开启存储优化开关，包括显存垃圾及时回收和Op内部的输出复用输入等。若存储空间仍然不够，建议：

  1. 降低 `batch_size` ；
  2. 对index进行排序，减少padding的数量。

----------

##### 问题：训练过程中如果出现不收敛的情况，如何处理？

+ 答复：不收敛的原因有很多，可以参考如下方式排查：

  1. 检查数据集中训练数据的准确率，数据是否有错误，特征是否归一化；
  2. 简化网络结构，先基于benchmark实验，确保在baseline网络结构和数据集上的收敛结果正确；
  3. 对于复杂的网络，每次只增加一个改动，确保改动后的网络正确；
  4. 检查网络在训练数据上的Loss是否下降；
  5. 检查学习率、优化算法是否合适，学习率过大会导致不收敛；
  6. 检查`batch_size`设置是否合适，`batch_size`过小会导致不收敛；
  7. 检查梯度计算是否正确，是否有梯度过大的情况，是否为`NaN`。

----------

##### 问题：Loss为NaN，如何处理？

+ 答复：可能由于网络的设计问题，Loss过大（Loss为NaN）会导致梯度爆炸。如果没有改网络结构，但是出现了NaN，可能是数据读取导致，比如标签对应关系错误。还可以检查下网络中是否会出现除0，log0的操作等。

----------


##### 问题：训练后的模型很大，如何压缩？

+ 答复：建议您使用飞桨模型压缩工具[PaddleSlim](https://www.paddlepaddle.org.cn/tutorials/projectdetail/489539)。PaddleSlim是飞桨开源的模型压缩工具库，包含模型剪裁、定点量化、知识蒸馏、超参搜索和模型结构搜索等一系列模型压缩策略，专注于**模型小型化技术**。

----------

##### 问题：`load_inference_model`在加载预测模型时能否用`py_reader`读取？

+ 答复：目前`load_inference_model`加载进行的模型还不支持`py_reader`输入。

----------

##### 问题：预测时如何打印模型中每一步的耗时？  

+ 答复：可以在设置config时使用`config.enable_profile()`统计预测时每个算子和数据搬运的耗时。对于推理api的使用，可以参考官网文档[Python预测API介绍](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/python_infer_cn.html)。示例代码：
```python
# 设置config:
def set_config(args):
    config = Config(args.model_file, args.params_file)
    config.disable_gpu()
    # enable_profile()打开后会统计每一步耗时
    config.enable_profile()
    config.switch_use_feed_fetch_ops(False)
    config.switch_specify_input_names(True)
    config.switch_ir_optim(False)
    return config
```
----------

##### 问题：模型训练时如何进行梯度裁剪？

+ 答复：设置Optimizer中的`grad_clip`参数值。

----------

##### 问题：静态图模型如何拿到某个variable的梯度？

+ 答复：

 1. 使用`paddle.static.Print()`接口，可以打印中间变量及其梯度；
 2. 将变量梯度名放到fetch_list里，通过`exe.run()`获取，一般variable的梯度名是variable的名字加上 "@GRAD"。
 3. 对于参数（不适用于中间变量和梯度），还可以通过`scope.find_var()`接口，通过变量名字查找对应的tensor。

 后两者方法需要使用变量名，飞桨中变量的命名规则请参见[Name](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api_guides/low_level/program.html#api-guide-name) 。


```python
import paddle
import numpy as np

paddle.enable_static()
data = paddle.static.data('data', shape=[4, 2])
out = paddle.static.nn.fc(x=data, size=1, num_flatten_dims=1, name='fc')

loss = paddle.mean(out)
loss = paddle.static.Print(loss)  # 通过 Print 算子打印中间变量及梯度
opt = paddle.optimizer.SGD(learning_rate=0.01)
opt.minimize(loss)

exe = paddle.static.Executor()
exe.run(paddle.static.default_startup_program())
loss, loss_g, fc_bias_g = exe.run(
    paddle.static.default_main_program(),
    feed={'data': np.random.rand(4, 2).astype('float32')},
    fetch_list=[loss, loss.name + '@GRAD', 'fc.b_0@GRAD'])  # 通过将变量名加入到fetch_list获取变量

print(loss, loss_g, fc_bias_g)
print(paddle.static.global_scope().find_var('fc.b_0').get_tensor())  # 通过scope.find_var 获取变量
```
