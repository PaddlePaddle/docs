## torch.reciprocal
### [torch.reciprocal](https://pytorch.org/docs/stable/generated/torch.reciprocal.html?highlight=reciprocal#torch.reciprocal)

```python
torch.reciprocal(input, 
            *, 
            out=None)
```

### [paddle.reciprocal](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/reciprocal_cn.html#reciprocal)

```python
paddle.reciprocal(x, 
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
# tensor([-0.4595, -2.1219, -1.4314,  0.7298])
torch.reciprocal(a)
# 输出
# tensor([-2.1763, -0.4713, -0.6986,  1.3702])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.reciprocal(x)
print(out)
# 输出
# [-2.5        -5.         10.          3.33333333]
```
