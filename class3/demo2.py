from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

path = r"D:\学习资料\深度学习学习笔记\nlp\class3\test-trainer"

from transformers import TrainingArguments

#定义超参数 
#包含 Trainer 用于训练和评估的所有超参数
training_args = TrainingArguments(path,evaluation_strategy="epoch")

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

# sample1 = tokenized_datasets["train"][:8]
# sample2 = tokenized_datasets["validation"][:8]

# trainer = Trainer(
#     model,
#     training_args,
#     train_dataset=sample1,
#     eval_dataset=sample2,
#     data_collator=data_collator,
#     tokenizer=tokenizer,
# )
import evaluate

import numpy as np

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    
)

trainer.train()