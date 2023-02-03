## torch.cos
### [torch.cos](https://pytorch.org/docs/stable/generated/torch.cos.html?highlight=cos#torch.cos)

```python
torch.cos(input, 
            *, 
            out=None)
```

### [paddle.cos](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/cos_cn.html#cos)

```python
paddle.cos(x, 
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
# tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
torch.cos(a)
# 输出
# tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.cos(x)
print(out)
# 输出
# [0.92106099 0.98006658 0.99500417 0.95533649]
```
