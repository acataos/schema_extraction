from sentence_transformers import SentenceTransformer
modelPath = "sentence_transformers_model"

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
model.save(modelPath)
