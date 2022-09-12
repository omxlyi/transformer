import torch
from transformers import AutoModelForSequenceClassification


checkpoint = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(device)