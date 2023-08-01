## [torch 参数更多]torch.Tensor.to

### [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

```python
torch.Tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#cast-dtype)

```python
paddle.Tensor.cast(dtype)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| dtype         | dtype        | 表示输出 Tensor 的数据类型。                                            |
| non_blocking  | -            | 控制 cpu 和 gpu 数据的异步复制，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| copy          | -            | 表示是否复制，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。   |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

---

### [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

```python
torch.Tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#cast-dtype)

```python
paddle.Tensor.cast(dtype)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| device        | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要转写。               |
| dtype         | dtype        | 表示输出 Tensor 的数据类型。                                            |
| non_blocking  | -            | 控制 cpu 和 gpu 数据的异步复制，Paddle 无此参数，暂无转写方式。                   |
| copy          | -            | 表示是否复制，Paddle 无此参数，暂无转写方式。                                     |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### device: Tensor 的设备

```python
# Pytorch 写法
y = x.to(device=torch.device('cpu'), dtype=torch.float64)

# Paddle 写法
y = x.cast(paddle.float64)
y.cpu()
```

---

### [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

```python
torch.Tensor.to(other, non_blocking=False, copy=False)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/Tensor_cn.html#cast-dtype)

```python
paddle.Tensor.cast(dtype)
```

Pytorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                      |
| ------------ | ------------ | --------------------------------------------------------- |
| other        | -            | 表示参照 dtype 的 Tensor，Paddle 无此参数，需要转写。 |
| non_blocking | -            | 控制 cpu 和 gpu 数据的异步复制，Paddle 无此参数，暂无转写方式。     |
| copy         | -            | 表示是否复制，Paddle 无此参数，暂无转写方式。                       |

### 转写示例

#### other: 表示参照 dtype 的 Tensor

```python
# Pytorch 写法
y = x.to(x2)

# Paddle 写法
y = x.cast(x2.dtype)
```
