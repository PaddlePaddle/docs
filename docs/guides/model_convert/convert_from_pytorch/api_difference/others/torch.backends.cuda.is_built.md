## [ 组合替代实现 ]torch.backends.cuda.is_built

### [torch.backends.cuda.is_built]https://pytorch.org/docs/stable/backends.html?highlight=torch+backends+cudnn+is_available#torch.backends.cuda.is_built)
```python
torch.backends.cuda.is_built()
```

检测是否在 cuda 环境下编译安装包。

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

###  转写示例
```python
# Pytorch 写法
torch.backends.cuda.is_built()

# Paddle 写法
not 'False' in paddle.version.cuda()
```
