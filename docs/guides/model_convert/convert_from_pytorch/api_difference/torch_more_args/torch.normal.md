## [ torch 参数更多 ]torch.normal
### [torch.normal](https://pytorch.org/docs/stable/generated/torch.normal.html#torch.normal)
```python
torch.normal(mean,
             std,
             *,
             generator=None,
             out=None)
```
### [paddle.normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/normal_cn.html)
```python
paddle.normal(mean=0.0,
              std=1.0,
              shape=None,
              name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| mean          | mean         | 表示正态分布的均值。                                     |
| std           | std          | 表示正态分布的方差。                                     |
| -             | shape        | 表示输出 Tensor 的形状，PyTorch 无此参数， Paddle 保持默认即可。                                     |
| generator     | -            | 用于采样的伪随机数生成器，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| out           | -            | 表示输出的 Tensor， Paddle 无此参数，需要转写。               |


### 转写示例
#### out：指定输出
```python
# PyTorch 写法
torch.normal(torch.tensor([0, 0, 0]), torch.tensor([1.0, 2.0, 3.0]), out=y)

# Paddle 写法
paddle.assign(paddle.normal(paddle.to_tensor([0, 0, 0]), paddle.to_tensor([1.0, 2.0, 3.0])), y)
```
