##  [ 输入参数用法不一致 ]torch.nn.functional.smooth_l1_loss

### [torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html)

```python
torch.nn.functional.smooth_l1_loss(input,
                    target,
                    size_average=None,
                    reduce=None,
                    reduction='mean',
                    beta=1.0)
```

### [paddle.nn.functional.smooth_l1_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/nn/functional/smooth_l1_loss_cn.html#smooth-l1-loss)

```python
paddle.nn.functional.smooth_l1_loss(input,
                    label,
                    reduction='mean',
                    delta=1.0,
                    is_huber=True,
                    name=None)
```

两者功能一致，但 Paddle 的 `delta` 和 PyTorch 的 `beta` 参数在公式中用法不一致，具体如下：

### 参数映射

| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 输入 Tensor                                     |
| target          | label         | 输入 Tensor 对应的标签，仅参数名不一致。                                |
| size_average          | -         | 已弃用                                      |
| reduce          | -         | 已弃用                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'   |
| beta          | delta         | SmoothL1Loss 损失的阈值参数                       |
| -          | is_huber         | 控制 huber_loss 与 smooth_l1_loss 的开关，Paddle 需设置为 False 。                     |

Torch 中 Smooth L1 loss 的计算方式:

$$
\ell(x, y) = \left [l_1, ..., l_N\ \right ]^T
$$

其中:

$$
l_n = \begin{cases}
0.5 (x_n - y_n)^2 / beta, & \text{if } |x_n - y_n| < beta \\
|x_n - y_n| - 0.5 * beta, & \text{otherwise }
\end{cases}
$$

而 Paddle 中 Smooth L1 loss 的计算方式:

$$
loss(x,y)  = \left [ z_1, ..., z_N \right ]^T
$$

其中：

$$
z_i = \begin{cases}
        0.5(x_i - y_i)^2 & {if |x_i - y_i| < delta} \\
        delta * |x_i - y_i| - 0.5 * delta^2 & {otherwise}
        \end{cases}
$$

当 $is\_huber$ 参数为 True 时，Paddle 为 huber 损失。
当 $is\_huber$ 参数为 False 时，PyTorch 和 Paddle 计算过程一致，均为 huber 损失除以 $delta$ 值。


### 转写示例

#### size_average


```python
# PyTorch 的 size_average、 reduce 参数转为 Paddle 的 reduction 参数
if size_average is None:
    size_average = True
if reduce is None:
    reduce = True
if size_average and reduce:
    reduction = 'mean'
elif reduce:
    reduction = 'sum'
else:
    reduction = 'none'
```

#### beta
```python
# PyTorch 的 beta 参数转化为 delta 参数
beta=0.8

# PyTorch 写法
output = torch.nn.functional.smooth_l1_loss(input, label, beta=beta)

# Paddle 写法
output = paddle.nn.functional.smooth_l1_loss(input, label, delta=beta, is_huber=False)
```
