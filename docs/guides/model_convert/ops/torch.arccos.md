## torch.arccos
### [torch.arccos](https://pytorch.org/docs/stable/generated/torch.arccos.html?highlight=arccos#torch.arccos)

```python
torch.arccos(input, 
                *, 
                out=None)
```

### [paddle.acos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/acos_cn.html#acos)

```python
paddle.acos(x, 
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
# 输出
# tensor([ 0.3348, -0.5889,  0.2005, -0.1584])
torch.arccos(a)
# 输出
# tensor([ 1.2294,  2.2004,  1.3690,  1.7298])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.acos(x)
print(out)
# 输出
# [1.98231317 1.77215425 1.47062891 1.26610367]
```
