from sentence_transformers import SentenceTransformer
import torch
from torch import cosine_similarity

#Loading a pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Testing in/out
resumes = ["I am a software engineer with 5 years of experience in Python and JavaScript.",
           "I am a data scientist with expertise in machine learning and statistics."]

job_descriptions = ["We are looking for a software engineer with experience in web development.",
                    "Looking for a data scientist with expertise in AI and data analytics."]

#Convert sentences to embeddings (vectors)
resume_embeddings = model.encode(resumes)
job_description_embeddings = model.encode(job_descriptions)

# Print the embeddings
print("Resume Embeddings:", resume_embeddings)
print("Job Description Embeddings:", job_description_embeddings)


# Convert the NumPy arrays to PyTorch tensors
resume_embeddings_tensor = torch.tensor(resume_embeddings)
job_description_embeddings_tensor = torch.tensor(job_description_embeddings)

# Calculate cosine similarity between resume and job description embeddings using PyTorch
similarities = cosine_similarity(resume_embeddings_tensor, job_description_embeddings_tensor)

# Print the similarity matrix
print("Cosine Similarity Matrix:")
print(similarities)