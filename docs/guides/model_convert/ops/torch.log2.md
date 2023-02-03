## torch.log2
### [torch.log2](https://pytorch.org/docs/stable/generated/torch.log2.html?highlight=log2#torch.log2)

```python
torch.log2(input,
            *,
            out=None)
```

### [paddle.log2](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/log2_cn.html#log2)

```python
paddle.log2(x,
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
a = torch.rand(5)
a
# 输出
# tensor([ 0.8419,  0.8003,  0.9971,  0.5287,  0.0490])
torch.log2(a)
# 输出
# tensor([-0.2483, -0.3213, -0.0042, -0.9196, -4.3504])
```

``` python
# PaddlePaddle 示例：
# example 1: x is a float
x_i = paddle.to_tensor([[1.0], [2.0]])
res = paddle.log2(x_i)
# 输出
# [[0.], [1.0]]

# example 2: x is float32
x_i = paddle.full(shape=[1], fill_value=2, dtype='float32')
paddle.to_tensor(x_i)
res = paddle.log2(x_i)
print(res)
# 输出
# [1.0]

# example 3: x is float64
x_i = paddle.full(shape=[1], fill_value=2, dtype='float64')
paddle.to_tensor(x_i)
res = paddle.log2(x_i)
print(res)
# 输出
# [1.0]
```
