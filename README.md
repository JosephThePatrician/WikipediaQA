# WikipediaQA
Python project for answering questions based on Wikipedia

Algorithm:
1. Extract nouns from question using spacy
2. Search wikipedia for nouns using wikipedia search engine
3. Determine the best search result by calculating sentence similarity between question and search result title
4. Use QA model to answer the question with context from found wikipedia page
