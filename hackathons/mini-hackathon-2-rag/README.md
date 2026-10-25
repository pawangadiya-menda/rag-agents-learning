# Mini Hackathon-2: Retrieval-Augmented Generation (RAG)

This project implements a RAG question-answering assistant using the hackathon's AST-1 and AST-2 assignment documents as its knowledge base.

## Open the Google Colab notebook

[Open the RAG hackathon notebook in Google Colab](https://colab.research.google.com/drive/1S55dfsOgCcva1TY6_N-1_23KJQR_gKlA?usp=sharing)

> The notebook is hosted in Google Colab. To run it, ensure the Google account you use has access to the notebook. Upload `AST-1.pdf` and `AST-2.pdf` when prompted.

## Pipeline

1. Load PDF documents and preserve source-page metadata.
2. Split pages into overlapping text chunks.
3. Generate embeddings with Sentence Transformers.
4. Index embeddings in FAISS.
5. Retrieve context using similarity search or Maximal Marginal Relevance (MMR).
6. Generate grounded answers with FLAN-T5 by default, with an optional OpenAI API route.
7. Evaluate with prepared questions and compare retrieval/chunking settings.
8. Explore answers through a Gradio interface.

## Technologies

- Python and Google Colab
- PyPDF
- Sentence Transformers
- FAISS
- Hugging Face Transformers / FLAN-T5
- Gradio
- Optional OpenAI API

## Run instructions

1. Open the Colab link above.
2. Run cells from top to bottom.
3. Upload `AST-1.pdf` and `AST-2.pdf`.
4. Follow the notebook's dependency/setup instructions if the runtime reports package conflicts.
5. Run the evaluation questions and test the Gradio app.

## Experiments

- Similarity search vs. MMR retrieval
- Chunk size and overlap
- Retrieved source passages and page references
- Seven prepared evaluation questions

This is a learning/hackathon baseline. Review retrieved passages and verify generated answers against the source documents before relying on them.
