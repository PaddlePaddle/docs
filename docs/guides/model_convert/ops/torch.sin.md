## torch.sin
### [torch.sin](https://pytorch.org/docs/stable/generated/torch.sin.html?highlight=sin#torch.sin)

```python
torch.sin(input, 
            *, 
            out=None)
```

### [paddle.sin](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sin_cn.html#sin)

```python
paddle.sin(x, 
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
# tensor([-0.5461,  0.1347, -2.7266, -0.2746])
torch.sin(a)
# 输出
# tensor([-0.5194,  0.1343, -0.4032, -0.2711])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.sin(x)
print(out)
# 输出
# [-0.38941834 -0.19866933  0.09983342  0.29552021]
```
