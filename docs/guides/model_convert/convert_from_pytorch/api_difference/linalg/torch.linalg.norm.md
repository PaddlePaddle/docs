## [ 参数不一致 ]torch.linalg.norm
### [torch.linalg.norm](https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch-linalg-norm)

```python
torch.linalg.norm(A, ord=None, dim=None, keepdim=False, *, out=None, dtype=None)
```

### [paddle.linalg.norm](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/linalg/norm_cn.html#norm)

```python
paddle.linalg.norm(x, p='fro', axis=None, keepdim=False, name=None)
```

两者功能一致但参数类型不一致，Pytorch 返回 named tuple，Paddle 返回 Tensor，需要转写。具体如下：
### 参数映射
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| A | x | 表示输入的 Tensor ，仅参数名不一致。  |
| ord | p | 表示范数(ord)的种类 ，仅参数名不一致。  |
| dim | axis | 表示使用范数计算的轴 ，仅参数名不一致。  |
| keepdim | keepdim | 表示是否在输出的 Tensor 中保留和输入一样的维度。  |
| out | - | 表示范数(ord)的种类 ，仅参数名不一致。  |
| dtype | - | 表示输入 Tensor 转化的类型 ，paddle 无此参数，需要进行转写。  |


### 转写示例
#### 返回类型不一致
```python
# Pytorch 写法
torch.linalg.norm(x, dim=1)

# Paddle 写法
paddle.linalg.norm(x, axis=1)
```
