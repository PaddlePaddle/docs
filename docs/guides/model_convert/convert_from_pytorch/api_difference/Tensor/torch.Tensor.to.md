## [torch 参数更多]torch.Tensor.to

### [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

```python
torch.Tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#to-args-kwargs)

```python
paddle.Tensor.to(dtype, blocking=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| dtype         | dtype        | 表示输出 Tensor 的数据类型。                                            |
| non_blocking  | blocking     | 控制 cpu 和 gpu 数据的异步复制，取值相反，需要转写。                    |
| copy          | -            | 表示是否复制，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### non_blocking: cpu 和 gpu 数据的异步复制

``` python
# PyTorch 写法
x = torch.tensor([1, 2, 3])
x.to(dtype, non_blocking=True)

# Paddle 写法
x= paddle.to_tensor([1, 2, 3])
x.to(dtype, blocking=False)
```

---

### [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

```python
torch.Tensor.to(device=None, dtype=None, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#to-args-kwargs)

```python
paddle.Tensor.to(device, dtype=None, blocking=True)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                                    |
| ------------- | ------------ | ----------------------------------------------------------------------- |
| device        | device       | 表示 Tensor 存放设备位置。                                              |
| dtype         | dtype        | 表示输出 Tensor 的数据类型。                                            |
| non_blocking  | blocking     | 控制 cpu 和 gpu 数据的异步复制，取值相反，需要转写。                    |
| copy          | -            | 表示是否复制，Paddle 无此参数，暂无转写方式。                           |
| memory_format | -            | 表示内存格式，Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

#### non_blocking: cpu 和 gpu 数据的异步复制

``` python
# PyTorch 写法
x = torch.tensor([1, 2, 3])
x.to(device, dtype, non_blocking=True)

# Paddle 写法
x= paddle.to_tensor([1, 2, 3])
x.to(device, dtype, blocking=False)
```

---

### [torch.Tensor.to](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

```python
torch.Tensor.to(other, non_blocking=False, copy=False)
```

### [paddle.Tensor.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Tensor_cn.html#to-args-kwargs)

```python
paddle.Tensor.to(dtype)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch      | PaddlePaddle | 备注                                                  |
| ------------ | ------------ | ----------------------------------------------------- |
| other        | -            | 表示参照 dtype 的 Tensor，Paddle 无此参数，需要转写。 |
| non_blocking | blocking     | 控制 cpu 和 gpu 数据的异步复制，取值相反，需要转写。  |
| copy         | -            | 表示是否复制，Paddle 无此参数，暂无转写方式。         |

### 转写示例

#### other: 表示参照 dtype 的 Tensor

```python
# PyTorch 写法
y = x.to(x2)

# Paddle 写法
y = x.to(x2.dtype)
```

#### non_blocking: cpu 和 gpu 数据的异步复制

``` python
# PyTorch 写法
y = x.to(x2, non_blocking=True)

# Paddle 写法
y = x.to(x2.dtype, blocking=False)
```
