## [torch 参数更多 ] torch.Tensor.to
torch.Tensor.to 根据不同类型参数调用不同重载函数，可分别使用 paddle.Tensor.cast 实现转写

----------------

### [torch.Tensor.to](https://pytorch.org/docs/2.0/generated/torch.Tensor.to.html#torch-tensor-to)

```python
torch.Tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cast_cn.html#cast)

```python
paddle.Tensor.cast(dtype)
```

两者功能类似，参数不一致，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| dtype     | dtype            | 输出 Tensor 的数据类型 |
| non_blocking   | -          | 用于控制 cpu 和 gpu 数据的异步复制，转写无需考虑该参数，可直接删除。 |
| copy  | -          | 表示是否创建一个新的张量，当 copy 设置`True`时，即使张量已经符合所需的转换，也会创建一个新的张量。转写无需考虑该参数，可直接删除。 |
| memory_format       | -          | 更表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

```python
# torch 写法
torch.Tensor.to(torch.float64)

# paddle 写法
paddle.Tensor.cast(dtype='float64')
```

----------------

### [torch.Tensor.to](https://pytorch.org/docs/2.0/generated/torch.Tensor.to.html#torch-tensor-to)

```python
torch.to(device=None, dtype=dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cast_cn.html#cast)

```python
paddle.Tensor.cast(dtype)
```

两者功能类似，参数不一致，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| device     | -            | 表示 Tensor 存放设备位置，Paddle 无此参数，需要进行转写。 |
| dtype     | dtype            | 表示输出 Tensor 的数据类型 |
| non_blocking   | -          | 用于控制 cpu 和 gpu 数据的异步复制，转写无需考虑该参数。 |
| copy  | -          | 用于创建新的 Tensor 复制，转写无需考虑该参数。 |
| memory_format       | -          | 更表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

```python
# torch 写法
x.to(device=torch.device('cuda:0'), dtype=torch.float64)

# paddle 写法
x.cast(dtype='float64')
```

----------------


### [torch.Tensor.to](https://pytorch.org/docs/2.0/generated/torch.Tensor.to.html#torch-tensor-to)

```python
torch.to(other, non_blocking=False, copy=False)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cast_cn.html#cast)

```python
paddle.Tensor.cast(dtype)
```

两者功能类似，参数不一致，具体如下：
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| other     | -           | 表示输出 Tensor 的类型需要对齐输出的 Tensor，Paddle 无此参数，需要进行转写。 |
| non_blocking   | -          | 用于控制 cpu 和 gpu 数据的异步复制，转写无需考虑该参数。 |
| copy  | -          | 用于创建新的 Tensor 复制，转写无需考虑该参数。 |
| -     | dtype            | 表示输出 Tensor 的数据类型，PyTorch 无此参数，Paddle 需根据 other 类型进行转写。|

### 转写示例

```python
# torch 写法
x = torch.tensor([1,2,3], dtype="int32")
y.to(other=x)

# paddle 写法
x = paddle.to_tensor(data=[1, 2, 3], dtype='int32')
y.cast(x.dtype)
```
