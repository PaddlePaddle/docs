## torch.multiply
### [torch.multiply](https://pytorch.org/docs/stable/generated/torch.multiply.html?highlight=multiply#torch.multiply)

```python
torch.multiply(input,
                other,
                *,
                out=None)
```

### [paddle.multiply](https://vpaddlepaddle.org.cn/documentation/docs/zh/api/paddle/multiply_cn.html#multiply)

```python
paddle.multiply(x,
                y,
                name=None)
```

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input         | x            | 输入的 Tensor。                                      |
| other         | y            | 输入的 Tensor。                                      |
| out           | -            | 表示输出的 Tensor，PaddlePaddle 无此参数。               |


### 代码示例
``` python
# PyTorch 示例：
a = torch.randn(3)
a
# 输出
# tensor([ 0.2015, -0.4255,  2.6087])
torch.multiply(a, 100)
# 输出
# tensor([  20.1494,  -42.5491,  260.8663])

b = torch.randn(4, 1)
b
# 输出
# tensor([[ 1.1207],
#        [-0.3137],
#        [ 0.0700],
#        [ 0.8378]])
c = torch.randn(1, 4)
c
# 输出
# tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])
torch.multiply(b, c)
# 输出
# tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],
#        [-0.1614, -0.0382,  0.1645, -0.7021],
#        [ 0.0360,  0.0085, -0.0367,  0.1567],
#        [ 0.4312,  0.1019, -0.4394,  1.8753]])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([[1, 2], [3, 4]])
y = paddle.to_tensor([[5, 6], [7, 8]])
res = paddle.multiply(x, y)
print(res)
# 输出
# [[5, 12], [21, 32]]

x = paddle.to_tensor([[[1, 2, 3], [1, 2, 3]]])
y = paddle.to_tensor([2])
res = paddle.multiply(x, y)
print(res)
# 输出
# [[[2, 4, 6], [2, 4, 6]]]
```
