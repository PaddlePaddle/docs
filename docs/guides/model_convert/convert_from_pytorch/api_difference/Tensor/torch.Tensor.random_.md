## [ 组合替代实现 ] torch.Tensor.random_

### [torch.Tensor.random_](https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html)

```python
torch.Tensor.random_(from=0, to=None, *, generator=None)
```

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

### 参数映射

| PyTorch   | PaddlePaddle | 备注                                                                                 |
| --------- | ------------ | ------------------------------------------------------------------------------------ |
| from      | -            | 均匀分布最小值，需要转写。                                                           |
| to        | -            | 均匀分布最大值，需要转写。                                                           |
| generator | -            | 用于采样的伪随机数生成器， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

```python
# PyTorch 写法
out = x.random_(from=0, to=10)

# Paddle 写法
out = paddle.cast(paddle.randint(low=0, high=10, shape=x.shape), dtype='float32')
x = out
```
