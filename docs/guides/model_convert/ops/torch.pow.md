## torch.pow
### [torch.pow](https://pytorch.org/docs/stable/generated/torch.pow.html?highlight=pow#torch.pow)

```python
torch.pow(input, 
            exponent, 
            *, 
            out=None)
```

### [paddle.pow](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/pow_cn.html#pow)

```python
paddle.pow(x, 
            y, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| exponent      | y            | 指数值。                                             |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 代码示例
``` python
# PyTorch示例：
a = torch.randn(4)
a
# 输出
# tensor([ 0.4331,  1.2475,  0.6834, -0.2791])
torch.pow(a, 2)
# 输出
# tensor([ 0.1875,  1.5561,  0.4670,  0.0779])
exp = torch.arange(1., 5.)
a = torch.arange(1., 5.)
a
# 输出
# tensor([ 1.,  2.,  3.,  4.])
exp
# 输出
# tensor([ 1.,  2.,  3.,  4.])
torch.pow(a, exp)
# 输出
# tensor([   1.,    4.,   27.,  256.])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([1, 2, 3], dtype='float32')

# example 1: y is a float or int
res = paddle.pow(x, 2)
print(res)
# 输出
# Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1., 4., 9.])
res = paddle.pow(x, 2.5)
print(res)
# 输出
# Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1.         , 5.65685415 , 15.58845711])

# example 2: y is a Tensor
y = paddle.to_tensor([2], dtype='float32')
res = paddle.pow(x, y)
print(res)
# 输出
# Tensor(shape=[3], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
#        [1., 4., 9.])
```
