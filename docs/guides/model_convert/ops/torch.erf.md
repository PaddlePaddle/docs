## torch.erf
### [torch.erf](https://pytorch.org/docs/stable/generated/torch.erf.html?highlight=erf#torch.erf)

```python
torch.erf(input, 
            *, 
            out=None)
```

### [paddle.erf](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/erf_cn.html#erf)

```python
paddle.erf(x, 
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
torch.erf(torch.tensor([0, -1., 10.]))
# 输出
# tensor([ 0.0000, -0.8427,  1.0000])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.erf(x)
print(out)
# 输出
# [-0.42839236 -0.22270259  0.11246292  0.32862676]
```
