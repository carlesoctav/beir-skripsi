from sentence_transformers import SentenceTransformer, models

max_seq_length = 256  # Student model max. lengths for inputs (number of word pieces)
student_model_name = "indolem/indobert-base-uncased"

word_embedding_model = models.Transformer(
    student_model_name, max_seq_length=max_seq_length
)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), pooling_mode="cls"
)
student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

student_model.save_to_hub(
    repo_name="carles-undergrad-thesis/indobert-base-cls",
    organization="carles-undergrad-thesis",
)