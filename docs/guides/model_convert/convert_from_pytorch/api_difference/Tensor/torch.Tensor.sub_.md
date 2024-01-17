## [ torch 参数更多 ]torch.Tensor.sub_
### [torch.Tensor.sub_](https://pytorch.org/docs/stable/generated/torch.Tensor.sub_.html)

```python
torch.Tensor.sub_(other, *, alpha=1)
```

### [paddle.Tensor.subtract_]()

```python
paddle.Tensor.subtract_(y)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| other         | y            | 表示减数的 Tensor，仅参数名不一致。  |
| alpha         | -            | 表示`other`的乘数，Paddle 无此参数，需要转写。Paddle 应设置 y = alpha * other。  |


### 转写示例
#### alpha：表示`other`的乘数
```python
# PyTorch 写法
x.sub_(y, alpha=2)

# Paddle 写法
x.subtract_(2 * y)

# 注：Paddle 直接将 alpha 与 y 相乘实现
```
