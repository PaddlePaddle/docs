## [ 仅 API 调用方式不一致 ]transformers.StoppingCriteriaList

### [transformers.StoppingCriteriaList](https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/generation/stopping_criteria.py#L503)

```python
transformers.StoppingCriteriaList(*args)

```

### [paddleformers.generation.StoppingCriteriaList](https://github.com/PaddlePaddle/PaddleFormers/blob/ca66f8dd619a6b2e17fa901042277501b2ed3230/paddleformers/generation/stopping_criteria.py#L72)

```python
paddleformers.generation.StoppingCriteriaList(*args)

```

两者功能一致，均为继承自 Python 原生 `list` 的容器类，用于管理多个停止条件。参数定义与用法完全一致。

### 转写示例

```python
# PyTorch 写法
from transformers import StoppingCriteriaList, MaxLengthCriteria

criteria = MaxLengthCriteria(max_length=20)
stopping_criteria = StoppingCriteriaList([criteria])


# Paddle 写法
from[paddleformers.generation import StoppingCriteriaList, MaxLengthCriteria

criteria = MaxLengthCriteria(max_length=20)
stopping_criteria = StoppingCriteriaList([criteria])

```
