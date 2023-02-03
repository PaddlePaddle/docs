## torch.add
### [torch.add](https)

```python
torch.add(input, 
            other, 
            *, 
            alpha=1, 
            out=None)
```

### [paddle.add](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/add_cn.html#add)

```python
paddle.add(x, 
            y, 
            name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                     |
| other         | y            | 输入的 Tensor。                                     |
| alpha         | -            | other的乘数，PaddlePaddle无此参数。                   |
| out           | -            | 表示输出的Tensor，PaddlePaddle无此参数。               |


### 功能差异

#### 使用方式
***PyTorch***：out = input + alpha * other。  
***PaddlePaddle***：out = x + y。

### 代码示例
``` python
# PyTorch示例：
a = torch.randn(4)
a
# 输出
# tensortensor([ 0.0202,  1.0985,  1.3506, -0.6056])
torch.add(a, 20)
# 输出
# tensortensor([ 20.0202,  21.0985,  21.3506,  19.3944])

b = torch.randn(4)
b
# 输出
# tensortensor([-0.9732, -0.3497,  0.6245,  0.4022])
c = torch.randn(4, 1)
c
# 输出
# tensortensor([[ 0.3743],
        [-1.7724],
        [-0.5811],
        [-0.8017]])
torch.add(b, c, alpha=10)
# 输出
# tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
        [-18.6971, -18.0736, -17.0994, -17.3216],
        [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
        [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([2, 3, 4], 'float64')
y = paddle.to_tensor([1, 5, 2], 'float64')
z = paddle.add(x, y)
print(z)  
# 输出
# [3., 8., 6. ]
```
