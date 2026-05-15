Task 1: CNN
Model:
I built a simple CNN with 2 convolution layers, ReLU activation, and max pooling. Then I used fully connected layers for classification.
Approach:
I loaded the dataset using ImageFolder and applied resizing, normalization, and flipping. The model was trained using Adam optimizer and binary cross entropy loss. Finally, I tested the model on test data.
Accuracy:
The model achieved around 65% accuracy.


Task 2:RAG Chatbot
Approach :
I created a small structured CV and converted the text into embeddings using a SentenceTransformer model. These embeddings were stored in FAISS.
When a user asks a question, the system finds the most relevant part of the CV and returns it as the answer.
