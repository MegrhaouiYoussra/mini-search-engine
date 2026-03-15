# Mini Search Engine

A simple search engine implemented in Python using an inverted index and TF-IDF ranking algorithm.

This project demonstrates fundamental concepts used in information retrieval systems and search engines such as tokenization, stopword removal, inverted indexing, and document ranking.

## Features

- Text tokenization and normalization
- Stopword removal
- Inverted index using hash maps
- TF-IDF ranking
- Multi-word search queries
- Ranked search results

## Technologies

- Python
- Data Structures
- Hash Maps
- TF-IDF ranking
- Information Retrieval concepts

## How It Works

1. Documents are processed and tokenized.
2. Common stopwords are removed.
3. An inverted index is built mapping words to documents.
4. Queries are tokenized and matched against the index.
5. Documents are ranked using TF-IDF scoring.

## Example Query

machine learning neural networks

Example output:

1. [doc 1] score=0.42  
Machine learning algorithms include decision trees...

2. [doc 3] score=0.31  
Neural networks are inspired by the human brain...

## How to Run

Run the program using Python:

python search_engine.py

## Project Structure

mini-search-engine
│
├── search_engine.py
└── README.md

## Concepts Demonstrated

- Inverted Index
- TF-IDF Ranking
- Text Processing
- Hash Maps
- Information Retrieval
