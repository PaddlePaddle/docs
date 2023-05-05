## [部分参数不一致]torch.cuda.device

### [torch.cuda.device](https://pytorch.org/docs/1.13/generated/torch.cuda.device.html)

```python
    torch.cuda.device(device)
```

### [paddle.Tensor.transpose](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/CUDAPlace_cn.html#cudaplace)

```python
    paddle.CUDAPlace(id)
```

### 不一致的参数
两者部分参数用法不同，具体如下：
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| device | id | torch 的 device 参数类型为 torch.device 或 int 。paddle 的 id 为 int。 |

# 代码转写

```python
    # pytorch
    device = torch.cuda.device(0)

    # paddle
    device = paddle.CUDAPlace(0)
```
