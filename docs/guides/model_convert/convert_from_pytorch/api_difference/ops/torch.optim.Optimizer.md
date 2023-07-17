## [ 参数不一致 ]torch.optim.Optimizer

### [torch.optim.Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)

```python
torch.optim.Optimizer(params, defaults)
```

### [paddle.optimzier.Optimizer](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Optimizer_cn.html#paddle.optimizer.Optimizer)

```python
paddle.optimizer.Optimizer(learning_rate=0.001, epsilon=1e-08, parameters=None, weight_decay=None, grad_clip=None, name=None)
```

其中 Pytorch 的 defaults 与 Paddle 的参数用法不一致，具体如下：

### 参数映射

|  PyTorch   | PaddlePaddle |        备注        |
|  --------  |  ----------  |  ----------------  |
| params |  parameters  | 指定优化器需要优化的参数。 |
| defaults |  -  | 包含优化选项默认值的字典。Paddl
e 无此参数，需要进行转写。 |
| - |  learning_rate  | 学习率，用于参数更新的计算。 |
| - |  epsilon  | 添加到分母的一个很小值，避免发生除零错误。 |
| - |  weight_decay  | 正则化方法。 |
| - |  grad_clip  | 梯度裁剪的策略。 |


### 转写示例
#### defaults：包含优化选项默认值的字典
```python
# Pytorch 写法
torch.optim.Optimizer(params, defaults)

# Paddle 写法
paddle.optimizer.Optimizer(parameters=params, **defaults)
```
