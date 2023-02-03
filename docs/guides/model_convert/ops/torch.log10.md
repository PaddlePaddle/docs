## torch.log10
### [torch.log10](https://pytorch.org/docs/stable/generated/torch.log10.html?highlight=log10#torch.log10)

```python
torch.log10(input, 
            *, 
            out=None)
```

### [paddle.log10](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log10_cn.html#log10)

```python
paddle.log10(x, 
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
a = torch.rand(5)
a
# 输出
# tensor([ 0.5224,  0.9354,  0.7257,  0.1301,  0.2251])
torch.log10(a)
# 输出
# tensor([-0.2820, -0.0290, -0.1392, -0.8857, -0.6476])

```

``` python
# PaddlePaddle示例：
# example 1: x is a float
x_i = paddle.to_tensor([[1.0], [10.0]])
res = paddle.log10(x_i) 
# 输出
# [[0.], [1.0]]

# example 2: x is float32
x_i = paddle.full(shape=[1], fill_value=10, dtype='float32')
paddle.to_tensor(x_i)
res = paddle.log10(x_i)
print(res) 
# 输出
# [1.0]

# example 3: x is float64
x_i = paddle.full(shape=[1], fill_value=10, dtype='float64')
paddle.to_tensor(x_i)
res = paddle.log10(x_i)
print(res) 
# 输出
# [1.0]
```
