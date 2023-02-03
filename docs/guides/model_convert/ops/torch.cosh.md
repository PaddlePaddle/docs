## torch.cosh
### [torch.cosh](https://pytorch.org/docs/stable/generated/torch.cosh.html?highlight=cosh#torch.cosh)

```python
torch.cosh(input, 
            *, 
            out=None)
```

### [paddle.cosh](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cosh_cn.html#cosh)

```python
paddle.cosh(x, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
a = torch.randn(4)
a
# 输出
# tensor([ 0.1632,  1.1835, -0.6979, -0.7325])
torch.cosh(a)
# 输出
# tensor([ 1.0133,  1.7860,  1.2536,  1.2805])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.cosh(x)
print(out)
# 输出
# [1.08107237 1.02006676 1.00500417 1.04533851]
```
