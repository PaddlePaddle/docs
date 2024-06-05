## [ 输入参数用法不一致 ]torch.nn.Module.to

### [torch.nn.Module.to](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)

```python
torch.nn.Module.to(device=None, dtype=None, non_blocking=False)
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to(device=None, dtype=None, blocking=None)
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                                                                     |
| ------------ | ------------ | -------------------------------------------------------------------------------------------------------- |
| device       | device       | Tensor 设备类型，PyTorch 为 torch.device，Paddle 为字符串 cpu，gpu:x，xpu:x 或 Place 对象，需要转写。                    |
| dtype        | dtype        | Tensor 数据类型，PyTorch 为字符串或 PyTorch 数据类型，Paddle 为 字符串或 Paddle 数据类型，需要转写。 |
| non_blocking | blocking     | 是否同步或异步拷贝，PyTorch 和 Paddle 取值相反，需要转写。                                           |

### 转写示例

#### device 参数：Tensor 设备类型

```python
# PyTorch 写法:
module = torch.nn.Module()
module.to(device=torch.device("cuda:0"))

# Paddle 写法:
module = paddle.nn.Layer()
module.to(device="gpu:0")
```

#### dtype 参数：Tensor 数据类型

```python
# PyTorch 写法:
module = torch.nn.Module()
module.to(dtype=torch.float32)

# Paddle 写法:
module = paddle.nn.Layer()
module.to(dtype=paddle.float32)
```

#### non_blocking 参数：是否同步或异步拷贝

```python
# PyTorch 写法:
module = torch.nn.Module()
module.to(non_blocking = False)

# Paddle 写法:
module = paddle.nn.Layer()
module.to(blocking=True)
```

---

### [torch.nn.Module.to](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)

```python
torch.nn.Module.to(tensor, non_blocking=False)
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to(device=None, dtype=None, blocking=None)
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                           |
| ------------ | ------------ | -------------------------------------------------------------- |
| tensor       | -            | 获取设备和数据类型的 Tensor，Paddle 无此参数，需要转写。   |
| -            | device       | Tensor 设备类型，PyTorch 无此参数，需要转写。              |
| -            | dtype        | Tensor 数据类型，PyTorch 无此参数，需要转写。              |
| non_blocking | blocking     | 是否同步或异步拷贝，PyTorch 和 Paddle 取值相反，需要转写。 |

### 转写示例

#### tensor 参数：获取设备和数据类型的 Tensor

```python
# PyTorch 写法:
module = torch.nn.Module()
module.to(x)

# Paddle 写法:
module = paddle.nn.Layer()
module.to(device=x.place, dtype=x.dtype)
```

#### non_blocking 参数：是否同步或异步拷贝

```python
# PyTorch 写法:
module = torch.nn.Module()
module.to(x, non_blocking = False)

# Paddle 写法:
module = paddle.nn.Layer()
module.to(device=x.place, dtype=x.dtype, blocking=True)
```

---

### [torch.nn.Module.to](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.to)

```python
torch.nn.Module.to(memory_format=torch.channels_last)
```

memory_format 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除
