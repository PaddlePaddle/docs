## torch.normal
### [torch.normal](https://pytorch.org/docs/stable/generated/torch.normal.html?highlight=normal#torch.normal)
```python
torch.normal(mean,
             std,
             size=None,
             *,
             generator=None,
             out=None)
```
### [paddle.normal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/normal_cn.html#normal)
```python
paddle.normal(mean=0.0,
              std=1.0,
              shape=None,
              name=None)
```

其中 Pytorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| size          | shape        | 表示输出 Tensor 的形状。                                     |
| generator     | -            | 用于采样的伪随机数生成器，PaddlePaddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 转写示例
#### out：指定输出
```python
# Pytorch 写法
torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1), out=y)

# Paddle 写法
y = paddle.normal(mean=paddle.arange(1., 11.), std=paddle.arange(1, 0, -0.1))
```
