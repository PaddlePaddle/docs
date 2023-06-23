## [torch 参数更多]torch.Tensor.subtract_

### [torch.Tensor.subtract_](https://pytorch.org/docs/1.13/generated/torch.Tensor.subtract_.html#torch.Tensor.subtract_)

```python
torch.Tensor.subtract_(other, *, alpha=1)
```

### [paddle.Tensor.subtract_](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#id20)

```python
paddle.Tensor.subtract_(y, name=None)
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                         |
| ------- | ------------ | ------------------------------------------------------------ |
| other   | y            | 表示减数的 Tensor，仅参数名不一致。                          |
| alpha   | -            | 表示`other`的乘数，PaddlePaddle 无此参数，需要进行转写。Paddle 应设置 y = alpha * other。 |

### 转写示例

#### alpha：表示`other`的乘数
```python
# Pytorch 写法
x.subtract_(y, alpha=2)

# Paddle 写法
x.subtract_(2 * y)

# 注：Paddle 直接将 alpha 与 y 相乘实现
```
