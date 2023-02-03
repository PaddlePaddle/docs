## torch.fmod
### [torch.fmod](https://pytorch.org/docs/stable/generated/torch.fmod.html?highlight=fmod#torch.fmod)

```python
torch.fmod(input, 
            other, 
            *, 
            out=None)
```

### [paddle.mod](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/mod_cn.html#mod)

```python
paddle.mod(x, 
            y, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 被除数。                                              |
| other         | y            | 除数。                                               |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
torch.fmod(torch.tensor([-3., -2, -1, 1, 2, 3]), 2)
# 输出
# tensor([-1., -0., -1.,  1.,  0.,  1.])
torch.fmod(torch.tensor([1, 2, 3, 4, 5]), -1.5)
# 输出
# tensor([1.0000, 0.5000, 0.0000, 1.0000, 0.5000])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([2, 3, 8, 7])
y = paddle.to_tensor([1, 5, 3, 3])
z = paddle.remainder(x, y)
print(z)  
# 输出
# [0, 3, 2, 1]
```
