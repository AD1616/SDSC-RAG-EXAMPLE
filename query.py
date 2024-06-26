from langchain_community.vectorstores.chroma import Chroma
from embedding_functions import get_hg_embedding_function
from generate_response import generate
import argparse

CHROMA_PATH = "chroma"


def retrieve_relevant_docs(query_text: str):
    embedding_function = get_hg_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search(query_text, k=3)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    docs = retrieve_relevant_docs(query_text)
    response = generate(docs, query_text)
    print(response)


if __name__ == "__main__":
    main()