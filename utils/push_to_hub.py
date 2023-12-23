from transformers import AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("model/bertcat")
model = AutoModel.from_pretrained("model/bertcat")


model.push_to_hub("carles-undergrad-thesis/bertcat")
tokenizer.push_to_hub("carles-undergrad-thesis/bertcat")


