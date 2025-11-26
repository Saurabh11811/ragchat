from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from langchain import ConversationChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from flask import Flask, request, send_from_directory
from langchain.document_loaders import PyPDFLoader
import os, glob
import pandas as pd
from langchain.schema import Document

app = Flask(__name__)

datafolder = 'data'

def list_files_glob(extension):
    search_pattern = os.path.join(datafolder, '**', f'*.{extension}')
    files = glob.glob(search_pattern, recursive=True)
    return files

def load_excel_documents(file_path):
    documents = []
    xls = pd.ExcelFile(file_path, engine='openpyxl')
    for sheet_name in xls.sheet_names:
        sheet_df = pd.read_excel(xls, sheet_name=sheet_name, engine='openpyxl')
        for idx, row in sheet_df.iterrows():
            row_data = row.to_string(index=False)
            document_text = f"Sheet: {sheet_name}\nRow {idx + 1}:\n{row_data}"
            document = Document(page_content=document_text, metadata={"source": file_path, "sheet_name": sheet_name, "row_index": idx})
            documents.append(document)
    return documents

# Load the Falcon 7B model and tokenizer
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Wrap the model in a LangChain pipeline
pipeline = HuggingFacePipeline(pipeline=model, tokenizer=tokenizer)

# Initialize the LangChain LLM with the pipeline
llm = pipeline

# For embeddings, use a suitable open-source model or keep using OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Load and process documents
loaded_documents = []
pdf_files = list_files_glob('pdf')
for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)
    loaded_documents.extend(pdf_loader.load())

excel_files = list_files_glob('xlsx')
for excel_file in excel_files:
    loaded_documents.extend(load_excel_documents(excel_file))

# Convert documents into embeddings
chunked_documents = [Document(page_content=doc.page_content, metadata=doc.metadata) for doc in loaded_documents]
db = FAISS.from_documents(chunked_documents, embeddings)

# Setup memory for conversational context
memory = ConversationBufferWindowMemory(k=5, return_messages=True)

# Create a conversational retrieval chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    verbose=True,
    return_source_documents=True
)

@app.route('/')
def chatbot():
    return send_from_directory('', 'chatbot.html')

@app.route('/send')
def getresponse():
    query = request.args.get('query', '')
    print("Query Received: ", query)
    response = conversational_chain({"question": query})
    print(response)
    return response["answer"]

@app.route('/refresh')
def refresh():
    loaded_documents.clear()
    memory.clear()

    for pdf_file in list_files_glob('pdf'):
        pdf_loader = PyPDFLoader(pdf_file)
        loaded_documents.extend(pdf_loader.load())

    for excel_file in list_files_glob('xlsx'):
        loaded_documents.extend(load_excel_documents(excel_file))

    print("Data refreshed")
    return ("Data refreshed")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
