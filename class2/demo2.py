from transformers import BertModel

# model = BertModel.from_pretrained("bert-base-cased")

path = r"D:\学习资料\深度学习学习笔记\nlp\class2\model"

#加载本地模型
model = BertModel.from_pretrained(path)
print(model.config.id2label)

#保存模型
model.save_pretrained(path)


