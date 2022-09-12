# 零样本分类
from transformers import pipeline

classifier = pipeline("zero-shot-classification")


sentence = "This is a course about the Transformers library" #准备句子

labels = ["education", "politics", "business"] #准备标签

out = classifier(sentence,candidate_labels=labels)

print(out)

# 下载的一些东西
# config.json               构建模型体系结构所需的属性
# pytorch_model.bin         模型的所有权重
# tokenizer_config.json
# vocab.json
# merges.txt
# tokenizer.json

# 结果
# {'sequence': 'This is a course about the Transformers library', 
# 'labels': ['education', 'business', 'politics'], 
# 'scores': [0.8445993065834045, 0.11197393387556076, 0.043426718562841415]}