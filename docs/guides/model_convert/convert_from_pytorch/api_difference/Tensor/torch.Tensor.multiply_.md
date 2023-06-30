## [ 参数不一致 ] torch.Tensor.multiply_

### [torch.Tensor.multiply_](https://pytorch.org/docs/1.13/generated/torch.Tensor.multiply_.html?highlight=multiply#torch.Tensor.multiply_)

```python
torch.Tensor.multiply_(value)
```

### [paddle.Tensor.multiply](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#multiply-y-axis-1-name-none)

```python
paddle.Tensor.multiply(y,
                axis=-1,
                name=None)
```

其中，Paddle 与 PyTorch 的 `other` 参数所支持类型不一致，具体如下：

### 参数映射
| PyTorch       | PaddlePaddle | 备注                                             |
| ------------- | ------------ | ----------------------------------------------- |
| other         | y            | 相乘的元素，PyTorch 支持 Tensor 和 Python Number，Paddle 仅支持 Tensor，需要进行转写。                       |
| -             | axis         | 计算的维度，PyTorch 无此参数， Paddle 保持默认即可。|

### 转写示例
#### other：相乘的元素
```python
# PyTorch 写法
y = x.multiply_(other=2)

# Paddle 写法
y = x.multiply(y=paddle.to_tensor(2))
```
