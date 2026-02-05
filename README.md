Job Recommender System using NLP: This project is a smart job recommendation system that suggests relevant jobs based on a user’s skills. Instead of matching exact keywords, it understands the meaning of job descriptions and user input to find better matches.
The system converts job descriptions into numerical representations (embeddings) and compares them with the user’s skill input. It then uses FAISS to quickly find the most similar jobs and displays them through a web interface built with Flask.
What it does
Takes user skills as input
Processes job descriptions using NLP
Finds similar jobs using semantic search
Shows recommended jobs on a web page
Technologies used
Python
Flask
Sentence Transformers
FAISS
Pandas
