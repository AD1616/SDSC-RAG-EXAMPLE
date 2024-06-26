from langchain_community.vectorstores.chroma import Chroma
from embedding_functions import get_hg_embedding_function
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

CHROMA_PATH = "chroma"
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

"""
Takes in a list of documents and a string question.

Outputs which of those documents are relevant to the question and which are not, both as lists.

Replace <api_key> with the actual key.
"""
def generate(documents: list[Document], question: str):
    inference_server_url = "https://sdsc-llama3-api.nrp-nautilus.io/v1"

    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key="<api_key>",
        openai_api_base=inference_server_url,
        temperature=0,
    )

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Answer the question based only on the context.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the context: \n\n {context} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "context"],
    )

    retriever = prompt | llm

    context_text = ""
    for document in documents:
        context_text += f"\n\n {document.page_content} {document.metadata} \n\n"

    response = retriever.invoke({"question": question, "context": context_text}).content

    return response


if __name__ == "__main__":
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_hg_embedding_function())

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    question = "attention"

    documents = retriever.invoke(question)

    response = generate(documents, question)

    print(response)