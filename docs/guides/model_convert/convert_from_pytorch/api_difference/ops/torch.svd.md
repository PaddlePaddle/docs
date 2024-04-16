## [ torch 参数更多 ] torch.svd

### [torch.svd](https://pytorch.org/docs/stable/generated/torch.svd.html?highlight=torch+svd#torch.svd)

```python
torch.svd(input, some=True, compute_uv=True, *, out=None)
```

### [paddle.linalg.svd](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/linalg/svd_cn.html#svd)

```python
paddle.linalg.svd(x, full_matrics=False, name=None)
```

PyTorch 相比 Paddle 支持更多其他参数，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | x            | 输入 Tensor ，仅参数名不一致。                           |
| some          | full_matrics            | 表示需计算的奇异值数目。 与 PyTorch 默认值不同，需要转写。                         |
| compute_uv   | -            | 表示是否计算 U 和 V 。Paddle 无此参数，暂无转写方式。            |
| out          | -            | 表示输出的 Tensor 元组。 Paddle 无此参数，需要转写。 |

### 转写示例
#### some：表示需计算的奇异值数目
```python
# PyTorch 写法
u, s, v = torch.svd(x, some = True )

# Paddle 写法
u, s, v = paddle.linalg.svd(x， full_matrics = False)
```
#### out：指定输出
```python
# PyTorch 写法
torch.svd(x, out=(u, s, v) )

# Paddle 写法
u, s, v = paddle.linalg.svd(x)
```
