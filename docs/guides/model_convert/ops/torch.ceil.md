## torch.ceil
### [torch.ceil](https://pytorch.org/docs/stable/generated/torch.ceil.html?highlight=ceil#torch.ceil)

```python
torch.ceil(input, 
            *, 
            out=None)
```

### [paddle.ceil](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/ceil_cn.html#ceil)

```python
paddle.ceil(x, 
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
# tensor([-0.6341, -1.4208, -1.0900,  0.5826])
torch.ceil(a)
# 输出
# tensor([-0., -1., -1.,  1.])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.ceil(x)
print(out)
# 输出
# [-0. -0.  1.  1.]
```
