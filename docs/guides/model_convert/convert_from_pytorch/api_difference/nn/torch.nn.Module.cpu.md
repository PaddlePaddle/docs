## [参数不一致]torch.nn.Module.cpu

### [torch.nn.Module.cpu](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.cpu)

```python
torch.nn.Module.cpu()
```

### [paddle.nn.Layer.to](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Layer_cn.html#to-device-none-dtype-none-blocking-none)

```python
paddle.nn.Layer.to("cpu")
```

两者参数用法不一致，具体如下：

### 参数映射

| PyTorch | PaddlePaddle | 备注                              |
| ------- | ------------ | --------------------------------- |
| -       | device       | Paddle 设置为 cpu，需要转写。 |

### 转写示例

#### device 参数：设备

```python
# PyTorch 写法:
module = torch.nn.Module()
module.cpu()

# Paddle 写法:
module = paddle.nn.Layer()
module.to("cpu")
```
