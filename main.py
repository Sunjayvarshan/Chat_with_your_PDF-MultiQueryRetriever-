from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough

llm = ChatOllama(model = "llama3.1", temperature=0.3)

#### 1) INGESTION OF DATA
#loading the docs
docs = PyPDFLoader("./data/career-in-ai.pdf").load() + PyPDFLoader("./data/attention-is-all-you-need.pdf").load()

#splitting the docs
text_split = RecursiveCharacterTextSplitter(chunk_size = 800 , chunk_overlap = 80)
chunks = text_split.split_documents(docs)

#### 2) VECTOR DATABASE
#adding to vector-database
vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = OllamaEmbeddings(model= "nomic-embed-text", show_progress= True),
    collection_name= "local-rag"
    
)

#question generation prompt

query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 2
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)

#revival 
retreiver = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(),
    llm,
    prompt= query_prompt
)

#RAG
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

#chain
chain = (
    {"context" : retreiver, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

response = chain.invoke("briefly explain, what is softmax")
print(response)