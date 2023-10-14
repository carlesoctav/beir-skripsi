from sentence_transformers import losses, models, SentenceTransformer

model = SentenceTransformer('output/indolem/indobert-base-uncased-v1-mmarco')

model.save_to_hub("st-indobert-mmarco-v1", "carles-undergrad-thesis")