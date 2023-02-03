## torch.rsqrt
### [torch.rsqrt](https://pytorch.org/docs/stable/generated/torch.rsqrt.html?highlight=rsqrt#torch.rsqrt)

```python
torch.rsqrt(input,
            *,
            out=None)
```

### [paddle.rsqrt](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/rsqrt_cn.html#rsqrt)

```python
paddle.rsqrt(x,
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
# tensor([-0.0370,  0.2970,  1.5420, -0.9105])
torch.rsqrt(a)
# 输出
# tensor([    nan,  1.8351,  0.8053,     nan])
```

``` python
# PaddlePaddle 示例：
x = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])
out = paddle.rsqrt(x)
print(out)
# 输出
# [3.16227766 2.23606798 1.82574186 1.58113883]
```
