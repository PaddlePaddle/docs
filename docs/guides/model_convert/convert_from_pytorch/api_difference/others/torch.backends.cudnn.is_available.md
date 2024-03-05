## [ 组合替代实现 ]torch.backends.cudnn.is_available

### [torch.backends.cudnn.is_available](https://pytorch.org/docs/stable/backends.html?highlight=torch+backends+cudnn+is_available#torch.backends.cudnn.is_available)
```python
torch.backends.cudnn.is_available()
```

检测当前 cudnn 是否可用。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例
```python
# PyTorch 写法
torch.backends.cudnn.is_available()

# Paddle 写法
bool(paddle.device.get_cudnn_version())
```
