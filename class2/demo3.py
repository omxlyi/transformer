from transformers import AutoTokenizer

# 分词器
#     输入：字符串
#     输出：  input_ids
#             token_type_ids
#             attention_mask


# 加载分词器
#从官网
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#从本地
path = r"D:\学习资料\深度学习学习笔记\nlp\class2\tokenizer"
tokenizer = AutoTokenizer.from_pretrained(path)


#保存分词器
# tokenizer.save_pretrained(path)

# tokenizer_config.json
# vocab.txt
# tokenizer.json


# 分词器测试

sentence = "hello,i am weiyin."
out = tokenizer(sentence) #包括两步：分词、to input_ids
print(out)

# 1、分词
tokens = tokenizer.tokenize(sentence)
print(tokens)
# 2、将tokens 转换成 模型需要的 input_ids
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# 将input_ids 转换成tokens
tokens = tokenizer.convert_ids_to_tokens(ids)
print(tokens)

# 解码，即input_ids to string
str = tokenizer.decode(ids)
print(str)






