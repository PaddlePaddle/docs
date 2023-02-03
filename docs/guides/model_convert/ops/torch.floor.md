## torch.floor
### [torch.floor](https://pytorch.org/docs/stable/generated/torch.floor.html?highlight=floor#torch.floor)

```python
torch.floor(input,
            *,
            out=None)
```

### [paddle.floor](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/floor_cn.html#floor)

```python
paddle.floor(x,
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 代码示例
``` python
# PyTorch 示例：
a = torch.randn(4)
a
# 输出
# tensor([-0.8166,  1.5308, -0.2530, -0.2091])
torch.floor(a)
# 输出
# tensor([-1.,  1., -1., -1.])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.floor(x)
print(out)
# 输出
# [-1. -1.  0.  0.]
```
