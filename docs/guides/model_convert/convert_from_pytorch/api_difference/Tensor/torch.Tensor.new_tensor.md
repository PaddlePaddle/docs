## [torch 参数更多]torch.Tensor.new_tensor

### [torch.Tensor.new_tensor](https://pytorch.org/docs/stable/generated/torch.Tensor.new_tensor.html#torch-tensor-new-tensor)

```python
torch.Tensor.new_tensor(data, *, dtype=None, device=None, requires_grad=False, layout=torch.strided, pin_memory=False)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/to_tensor_cn.html)

```python
paddle.to_tensor(data, dtype=None, place=None, stop_gradient=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                         |
| ------------- | ------------ | ------------------------------------------------------------ |
| data          | data         | 数据内容。 |
| dtype         | dtype        | 表示输出 Tensor 类型，如果没有指定，默认使用当前对象的 dtype，需要转写。    |
| device        | place         | 创建 tensor 的设备位置，仅参数名不一致。                       |
| requires_grad | stop_gradient | 表示是否计算梯度，两者参数功能相反，需要转写。      |

### 转写示例

#### dtype：数据类型

```python
# PyTorch 写法
y = x.new_tensor(data)

# Paddle 写法
y = paddle.to_tensor(data, dtype=x.dtype)
```

#### requires_grad：是否求梯度

```python
# PyTorch 写法
y = x.new_tensor(data, requires_grad=True)

# Paddle 写法
y = paddle.to_tensor(data, stop_gradient=False)
```
