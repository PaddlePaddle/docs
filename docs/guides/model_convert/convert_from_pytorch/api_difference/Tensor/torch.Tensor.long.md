## [torch 参数更多]torch.Tensor.long

### [torch.Tensor.long](https://pytorch.org/docs/1.13/generated/torch.Tensor.long.html#torch.Tensor.long)

```python
torch.Tensor.long(memory_format=torch.preserve_format)
```

### [paddle.Tensor.astype](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#astype-dtype)

```python
paddle.Tensor.astype('int64')
```

其中 PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| -             | dtype        | 转换数据类型，PyTorch 无此参数，Paddle 设置为 paddle.int64。            |

### 转写示例

#### dtype 参数：转换数据类型

```python
# PyTorch 写法:
y = x.long()

# Paddle 写法:
y = x.astype(paddle.int64)
```
