# PantryPilot
![Python](https://img.shields.io/badge/-Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

## About
NLP-based **recipe recommendation system** that matches grocery and pantry ingredients to recipes through a vectorized similarity search.

Designed for:
- **Speed**
- **Explainability** (why recipes are suggested)
- **Food-waste** reduction

## Problem
- Choosing what to cook with limited ingredients is oftentimes difficult and can lead to major food waste
- Although recipe platforms provide great recommendations to users, it often makes the assumption that they will shop for missing ingredients, instead of based on what people already have

## Solution
Model recipe selection through:
- Pantry items - treated as queries
- Recipes - treated as documents
- NLP vecotrization and similarity search
- Results optimized for pantry utilization

## Features
- NLP-based recipe matching
    - Tokenizes and vectorizes recipe ingredients
    - Uses cosine similarity to rank recipes (based on overlap)
    - Scales to **250K+ recipes** with vector search
- Explainability in recommendations (provided)
    - Overlapping ingredients
    - Similarity score
    - Pantry utilization percentage
- Pantry utilization (optimized)
    - Recipes are ranked by similarity and efficiency (how they use available ingredients)

## Architecture
- User pantry
- Text preprocessing
- TF-IDF vectorization
- Cosine similarity search
- Ranked recommendations (top similarity)
- Explainability and utilization scores

## How to Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```