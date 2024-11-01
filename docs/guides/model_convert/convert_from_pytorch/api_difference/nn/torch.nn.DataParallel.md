## [ torch 参数更多 ]torch.nn.DataParallel
### [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html?highlight=dataparallel#torch.nn.DataParallel)

```python
torch.nn.DataParallel(module,
                      device_ids=None,
                      output_device=None,
                      dim=0)
```

### [paddle.DataParallel](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/DataParallel_cn.html#dataparallel)

```python
paddle.DataParallel(layers,
                    strategy=None,
                    comm_buffer_size=25,
                    last_comm_buffer_size=1,
                    find_unused_parameters=False)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| module        | layers       | 需要通过数据并行方式执行的模型。  |
| device_ids    | -            | 表示训练在哪几块 GPU 上，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| output_device | -            | 表示结果输出在哪一块 GPU 上，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。  |
| dim           | -            | 表示哪一维度上的数据进行划分，Paddle 无此参数。  |
| -             | strategy     |  PaddlePaddle 即将废弃参数。 |
| -             | comm_buffer_size |  它是通信调用（如 NCCLAllReduce）时，参数梯度聚合为一组的内存大小（MB），PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | last_comm_buffer_size |  它限制通信调用中最后一个缓冲区的内存大小（MB），PyTorch 无此参数，Paddle 保持默认即可。 |
| -             | find_unused_parameters |  是否在模型 forward 函数的返回值的所有张量中，遍历整个向后图，PyTorch 无此参数，Paddle 保持默认即可。 |

### 转写示例

#### GPU id 参数：指定使用哪些 GPU 训练
``` python
# PyTorch 写法:
model = torch.nn.DataParallel(SimpleModel(), device_ids=[0, 1])

# Paddle 写法:
# 只能在启动代码时设置 GPU id，设置方式如下：
python -m paddle.distributed.launch –selected_gpus=0,1 demo.py
```
