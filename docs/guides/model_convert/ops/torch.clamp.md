## torch.clamp
### [torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html?highlight=clamp#torch.clamp)

```python
torch.clamp(input,
            min=None,
            max=None,
            *,
            out=None)
```

### [paddle.clip](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/clip_cn.html#clip)

```python
paddle.clip(x,
            min=None,
            max=None,
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
# tensor([-1.7120,  0.1734, -0.0478, -0.0922])
torch.clamp(a, min=-0.5, max=0.5)
# 输出
# tensor([-0.5000,  0.1734, -0.0478, -0.0922])

min = torch.linspace(-1, 1, steps=4)
torch.clamp(a, min=min)
# 输出
# tensor([-1.0000,  0.1734,  0.3333,  1.0000])
```

``` python
# PaddlePaddle 示例：
x1 = paddle.to_tensor([[1.2, 3.5], [4.5, 6.4]], 'float32')
out1 = paddle.clip(x1, min=3.5, max=5.0)
out2 = paddle.clip(x1, min=2.5)
print(out1)
# 输出
# [[3.5, 3.5]
# [4.5, 5.0]]
print(out2)
# 输出
# [[2.5, 3.5]
# [[4.5, 6.4]
```
