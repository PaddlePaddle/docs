##  [ 参数用法不一致 ]torch.nn.functional.smooth_l1_loss

### [torch.nn.functional.smooth_l1_loss](https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html)

```python
torch.nn.functional.smooth_l1_loss(input,
                    target,
                    size_average=None,
                    reduce=None,
                    reduction='mean',
                    beta=1.0)
```

### [paddle.nn.functional.smooth_l1_loss](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/functional/smooth_l1_loss_cn.html#smooth-l1-loss)

```python
paddle.nn.functional.smooth_l1_loss(input,
                    label,
                    reduction='mean',
                    delta=1.0,
                    name=None)
```

两者功能一致，但 Paddle 的 delta 和 PyTorch 的 beta 参数在公式中用法不一致，具体如下：

### 参数差异
| PyTorch       | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| input          | input         | 输入 Tensor                                     |
| target          | label         | 输入 Tensor 对应的标签值                                  |
| size_average          | -         | 已弃用                                      |
| reduce          | -         | 已弃用                                     |
| reduction          | reduction         | 表示应用于输出结果的规约方式，可选值有：'none', 'mean', 'sum'                         |
| beta          | delta         | SmoothL1Loss 损失的阈值参数                       |

Torch 中 Smooth L1 loss 的计算方式:

$$
\ell(x, y) = \{l_1, ..., l_N\}^T
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
        loss(x,y)  = \{z_1, ..., z_N\}^T
$$

其中：

$$
\mathop{z_i} = \left\{\begin{array}{rcl}
        0.5(x_i - y_i)^2 & & {if |x_i - y_i| < delta} \\
        delta * |x_i - y_i| - 0.5 * delta^2 & & {otherwise}
        \end{array} \right.
$$

所以如果 PyTorch 函数参数 $beta$ 与 Paddle 中的参数 $delta$ 取值相同，则 Paddle 的 loss 要再除以 $delta$ 值才能与 Torch 中的结果对齐。
