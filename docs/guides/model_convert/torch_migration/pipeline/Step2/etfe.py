import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizer as HFBertTokenizer
from transformers import DataCollatorWithPadding
from transformers import BertForSequenceClassification as HFBertForSequenceClassification
from torch.optim import AdamW


# 定义模型
model = HFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
# 定义Tokenizer，实现文本到ID的转化
tokenizer = HFBertTokenizer.from_pretrained("bert-base-uncased")

# 数据处理，从文本到处理好的数据，input_ids，token_type_ids
def preprocess_function(examples):
    result = tokenizer(
        examples["sentence"],
        padding=False,
        max_length=128,
        truncation=True,
        return_token_type_ids=True, )
    if "label" in examples:
        result["labels"] = [examples["label"]]
    return result
# 定义数据集
dataset = Dataset.from_csv("demo_sst2_sentence/demo.tsv", sep="\t")
dataset = dataset.map(
    preprocess_function,
    batched=False,
    remove_columns=dataset.column_names,
    desc="Running tokenizer on dataset", )
dataset.set_format(
    "np", columns=["input_ids", "token_type_ids", "labels"])
    
# 定义BatchSampler，组合batch，shuffle数据
sampler = torch.utils.data.SequentialSampler(dataset)
collate_fn = DataCollatorWithPadding(tokenizer)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    sampler=sampler,
    num_workers=0,
    collate_fn=collate_fn, )

# 定义loss
criterion = torch.nn.CrossEntropyLoss()
# 定义 optimizer 优化器
optimizer = AdamW(params=model.parameters(), lr=3e-5)

# 训练
for batch in data_loader:
    
    output = model(**batch).logits
    loss = criterion(output, batch['labels'].reshape(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
 