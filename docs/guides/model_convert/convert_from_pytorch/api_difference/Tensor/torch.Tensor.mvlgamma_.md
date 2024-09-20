<!--
 * @Description:
 * @Author: Xiao
 * @Date: 2024-09-21 16:03:47
 * @LastEditTime: 2024-09-23 19:33:10
 * @LastEditors: Xiao
-->
## [ 参数完全一致 ]torch.Tensor.mvlgamma_

### [torch.Tensor.mvlgamma_](https://pytorch.org/docs/stable/generated/torch.Tensor.mvlgamma_.html#torch-tensor-mvlgamma)

```python
torch.Tensor.mvlgamma_(p)
```

### [paddle.Tensor.multigammaln_](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/multigammaln__cn.html#multigammaln)

```python
paddle.Tensor.multigammaln_(p, name=None)
```

### 转写示例

```python
# PyTorch 写法
y = x.mvlgamma_(p)

# Paddle 写法
y = x.multigammaln_(p)
```
