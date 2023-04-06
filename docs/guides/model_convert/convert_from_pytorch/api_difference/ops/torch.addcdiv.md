## [ 组合替代实现 ]torch.addcdiv

### [torch.addcdiv](https://pytorch.org/docs/master/generated/torch.addcdiv.html#torch.addcdiv)
```python
torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None)
```

###  功能介绍
用于实现矩阵 `tensor1` 与矩阵 `tensor2` 相除，再加上输入 `input` ，公式为：

$ out =  input + value *  (tensor1 / tensor2) $

PaddlePaddle 目前无对应 API，可使用如下代码组合实现该 API。

```python
import paddle

def addcdiv(input, tensor1, tensor2, value=1, out=None):
    out = input + value * tensor1 / tensor2
    return out
```
