## [参数不一致]torch.nn.Module.cuda

### [torch.nn.Module.cuda](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.cuda)

```python
torch.nn.Module.cuda(device=None)
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to(device="gpu")
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                                                       |
| ------- | ------------ | ---------------------------------------------------------- |
| device  | device       | PyTorch 为设备编号，Paddle 为 gpu:设备编号，需要转写。 |

### 转写示例

#### device 参数：设备

```python
# PyTorch 写法:
module = torch.nn.Module()
module.cuda(0)

# Paddle 写法:
module = paddle.nn.Layer()
module.to("gpu:0")
```
