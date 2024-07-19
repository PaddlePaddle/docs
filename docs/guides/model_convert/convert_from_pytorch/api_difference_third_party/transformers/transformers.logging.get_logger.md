## [组合替代实现]transformers.logging.get_logger

### [transformers.logging.get_logger](https://github.com/huggingface/transformers/blob/d625294d79341662784495551abdf45e6cb9372f/src/transformers/utils/logging.py#L147)

```python
transformers.logging.get_logger(name: Optional[str] = None)
```

Paddle 无此 API，需要组合实现。

### 转写示例

```python
# Pytorch 写法
transformers.logging.get_logger()

# Paddle 写法
paddle.utils.try_import('logging').getLogger(name=__name__)
```
