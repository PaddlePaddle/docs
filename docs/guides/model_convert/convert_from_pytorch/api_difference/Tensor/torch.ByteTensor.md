## [ 仅 paddle 参数更多 ] torch.ByteTensor

### [torch.ByteTensor](https://pytorch.org/docs/stable/tensors.html)

```python
torch.ByteTensor(data)
```

### [paddle.to_tensor](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/to_tensor_cn.html#to-tensor)

```python
paddle.to_tensor(data, dtype=paddle.uint8, place="cpu", stop_gradient=True)
```

其中 Paddle 比 PyTorch 支持更多参数，具体如下：

### 参数映射

| PyTorch | PaddlePaddle  | 备注                                                      |
| ------- | ------------- | --------------------------------------------------------- |
| -       | dtype         | Tensor 的数据类型，Paddle 为 paddle.uint8，需要进行转写。 |
| -       | place         | Tensor 的设备，Paddle 为 cpu，需要进行转写。              |
| -       | stop_gradient | 是否梯度传导，PyTorch 无此参数，Paddle 保持默认即可。     |

### 转写示例

#### Paddle 参数转写

```python
# PyTorch 写法:
torch.ByteTensor([1, 2])

# Paddle 写法:
paddle.to_tensor([1, 2], place="cpu", dtype=paddle.uint8)
```
