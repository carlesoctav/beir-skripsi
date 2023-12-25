from transformers import AutoModelForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("model/bertcat")
model = AutoModelForSequenceClassification.from_pretrained("model/bertcat")


model.push_to_hub("carles-undergrad-thesis/Indobertcat")
tokenizer.push_to_hub("carles-undergrad-thesis/Indobertcat")


