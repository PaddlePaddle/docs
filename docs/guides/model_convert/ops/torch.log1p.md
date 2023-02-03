## torch.log1p
### [torch.log1p](https://pytorch.org/docs/stable/generated/torch.log1p.html?highlight=log1p#torch.log1p)

```python
torch.log1p(input, 
            *, 
            out=None)
```

### [paddle.log1p](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log1p_cn.html#log1p)

```python
paddle.log1p(x, 
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
a = torch.randn(5)
a
# 输出
# tensor([-1.0090, -0.9923,  1.0249, -0.5372,  0.2492])
torch.log1p(a)
# 输出
# tensor([    nan, -4.8653,  0.7055, -0.7705,  0.2225])
```

``` python
# PaddlePaddle示例：
data = paddle.to_tensor([[0], [1]], dtype='float32')
res = paddle.log1p(data)
# 输出
# [[0.], [0.6931472]]
```
