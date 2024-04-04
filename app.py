from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyPDFLoader
from flask import Flask, request, send_from_directory

app = Flask(__name__)


datafolder = 'data'
import os, glob
def list_pdf_files_glob():
    # Construct the search pattern
    search_pattern = os.path.join(datafolder, '**', '*.pdf')
    
    # Use glob to find all PDF files in the directory and subdirectories
    pdf_files = glob.glob(search_pattern, recursive=True)
    print(pdf_files)
    return pdf_files

FILE_LOADER_MAPPING = {
    "pdf": (PyPDFLoader, {}),
}


config = {
'max_new_tokens': 1024,
'repetition_penalty': 1.1,
'temperature': 0.8,
'top_k': 50,
'top_p': 0.9,
'stream': True,
'gpu_layers' : 1
#'threads': int(os.cpu_count() / 2)
}

llm = CTransformers(
    model = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    model_type="mistral",
    **config
)


model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'mps'} #mac M1 have mps (check system config on your system)
#model_kwargs = {'device': 'cpu'} - its generic for CPU on any machine but is damn slow. u can use cuda on intel/nvidia setup but change the param
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

loaded_documents = []
loaders = [PyPDFLoader(x) for x in list_pdf_files_glob()]
for loader in loaders:
    loaded_documents.extend(loader.load())

#loaded_documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30, length_function = len)
chunked_documents = text_splitter.split_documents(loaded_documents)
persist_directory = 'db'
db = FAISS.from_documents(chunked_documents, embeddings)

retriever = db.as_retriever(search_kwargs={"k":1})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, verbose=True)


app = Flask(__name__)
@app.route('/')
def chatbot():
    return send_from_directory('', 'chatbot.html')  


@app.route('/send')
def getresponse():
    query = request.args.get('query', '')
    print("Query Received: ", query)
    response = qa(query)
    print(response)
    lines = response['result'].split('\n')
    print(lines)
    return f"{lines}"

@app.route('/refresh')
def refresh():
    # TODO
    print("Data refreshed")
    return ("Data refreshed")




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
