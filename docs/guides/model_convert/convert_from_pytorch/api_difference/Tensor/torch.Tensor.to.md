## torch.Tensor.to
### [torch.Tensor.to](https://pytorch.org/docs/2.0/generated/torch.Tensor.to.html#torch-tensor-to)

```python
torch.Tensor.to(dtype, non_blocking=False, copy=False, memory_format=torch.preserve_format)
```

### [paddle.Tensor.cast](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/sigmoid_cn.html)

```python
paddle.cast(x, dtype)
```

两者功能类似，参数不一致，但 torch 是类成员方式，paddle 是 funtion 调用方式，具体如下： 
### 参数映射
| PyTorch | PaddlePaddle | 备注                        |
|---------|--------------|---------------------------|
| -     | x            | 表示输入的Tensor。 |
| dtype     | dtype            | 输出 Tensor 的数据类型 |
| non_blocking   | -          | 用于控制 cpu 和 gpu 数据的异步复制，转写无需考虑该参数。 |
| copy  | -          | 用于创新新的Tensor复制，转写无需考虑该参数。 |
| memory_format       | -          | 更表示内存格式， Paddle 无此参数，一般对网络训练结果影响不大，可直接删除。 |

### 转写示例

```python
# torch 写法
tensor.to(torch.float64)

# paddle 写法
tensor.cast(dtype='float64')
```
