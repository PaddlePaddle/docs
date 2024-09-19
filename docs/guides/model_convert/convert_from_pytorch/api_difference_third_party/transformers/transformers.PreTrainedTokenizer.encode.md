## [返回参数类型不一致]transformers.PreTrainedTokenizer.encode

### [transformers.PreTrainedTokenizer.encode](https://hf-mirror.com/docs/transformers/v4.42.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.encode)

```python
transformers.PreTrainedTokenizer.encode(text, **kwargs)
```

### [paddlenlp.transformers.PreTrainedTokenizer.encode](https://github.com/PaddlePaddle/PaddleNLP/blob/88d4b19bc6865fb28c11d2ce83d07c3b4b8dc423/paddlenlp/transformers/tokenizer_utils_base.py#L2369)

```python
paddlenlp.transformers.PreTrainedTokenizer.encode(text, **kwargs)
```

### 参数映射

| transformers  | PaddlePaddle | 备注                                                   |
| ------------- | ------------ | ------------------------------------------------------ |
| text          | text         | 输入的编码文本。  |
| 返回值         | 返回值        | PyTorch 返回 Tensor 类型，Paddle 返回类型为 `BatchEncoding`，是一种 dict-like 对象，key 包含 `input_ids`、`attention_mask` 等属性，需要转写。 |

### 转写示例

```python
# Pytorch 写法
transformers.PreTrainedTokenizer.encode(text)

# Paddle 写法
paddlenlp.transformers.PreTrainedTokenizer.encode(text)["input_ids"]
```
