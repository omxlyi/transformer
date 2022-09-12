from datasets import load_dataset
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"    #检查点

tokenizer = AutoTokenizer.from_pretrained(checkpoint)   #分词器

raw_datasets = load_dataset("glue", "mrpc") #数据集
print(raw_datasets) #数据集基本信息
# DatasetDict({
#     train: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 3668
#     })
#     validation: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 408
#     })
#     test: Dataset({
#         features: ['sentence1', 'sentence2', 'label', 'idx'],
#         num_rows: 1725
#     })
# })

#提取训练集
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset)
# Dataset({
#     features: ['sentence1', 'sentence2', 'label', 'idx'],
#     num_rows: 3668
# })


#查看 label 数字的含义
print(raw_train_dataset.features['label'])

# tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
# # print(tokenized_sentences_1['input_ids'][0])

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)

# tmp = raw_train_dataset.map(tokenize_function,batched=True)
# print(tmp)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]} #去掉不必要的列
# for i in samples:
#     print(i)

batch = data_collator(samples)

for k, v in batch.items():
    print(k,v.shape)