## torch.exp
### [torch.exp](https://pytorch.org/docs/stable/generated/torch.exp.html?highlight=exp#torch.exp)

```python
torch.exp(input, 
            *, 
            out=None)
```

### [paddle.exp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/exp_cn.html#exp)

```python
paddle.exp(x, 
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
torch.exp(torch.tensor([0, math.log(2.)]))
# 输出
# tensor([ 1.,  2.])
```

``` python
# PaddlePaddle示例：
x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
out = paddle.exp(x)
print(out)
# 输出
# [0.67032005 0.81873075 1.10517092 1.34985881]
```
