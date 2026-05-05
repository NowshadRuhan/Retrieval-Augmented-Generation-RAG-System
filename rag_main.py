
# RAG (Retrieval-Augmented Generation) main script
"""
1. Run pip install -r requirements.txt before executing this script.
Make sure your: 
version of below are correct. 
langchain==0.1.16 
langchain-community==0.0.38 
openai==0.28.1
"""
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI


"""
1. Load the document and split it into chunks
"""
print("Loading document...")
loader = PyMuPDFLoader("about-me.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} documents.")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)
all_splits = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(all_splits)}")

"""
Create a persistent Chroma vector database in the 'db' directory.
Use the MiniLM-L6-v2 embedding model running on CPU for text embeddings.
Convert all document chunks (all_splits) into vector embeddings.
Store embeddings and metadata inside the Chroma database for retrieval.
Print confirmation once the vector store is successfully created.
"""
persist_directory = 'db'
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs)
vectordb = Chroma.from_documents(documents=all_splits, 
                                 embedding=embedding, 
                                 persist_directory=persist_directory)

"""
Check the contents of the ChromaDB to ensure embeddings are stored correctly
"""
# import sqlite3
# conn = sqlite3.connect("db/chroma.sqlite3")
# cursor = conn.cursor()
# # List all tables
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()
# print("Tables:", tables)


"""
Load a local Llama‑2 7B Chat GGUF model using the LlamaCpp runtime.
Enable GPU acceleration with n_gpu_layers and set batch/context sizes.
Use f16_kv for faster inference and lower memory usage.
Stream generated tokens to the console via CallbackManager.
"""
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
    model_path="/Users/nowshad/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q2_K.gguf",
    n_gpu_layers=100,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=True,
)
print("LLM initialized.")



"""
Define two prompt templates: one generic and one specifically formatted for Llama models.
The Llama‑optimized prompt uses the <SYS> and [INST] structure required by Llama‑2/3 chat models.
Create a ConditionalPromptSelector that automatically chooses the correct prompt based on the LLM type.
If the active model is LlamaCpp, the system selects the Llama‑style prompt; otherwise it uses the default prompt.
This ensures consistent formatting and prevents prompt‑format errors across different LLM backends.
Retrieve the appropriate prompt for the currently loaded LLM instance.
The final 'prompt' object is then ready to be passed into an LLMChain or used for inference.
"""
from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> 
    You are a helpful assistant eager to assist with providing better Google search results.
    <</SYS>> 
    
    [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
            relevant, and concise:
            {question} 
    [/INST]""",
)
DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a helpful assistant eager to assist with providing better Google search results. \
        Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
        relevant, and concise: \
        {question}""",
)
QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=DEFAULT_SEARCH_PROMPT,
    conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)
prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
prompt


"""
Create an LLMChain that connects the selected prompt with the loaded Llama model.
Pass a question into the chain to generate an answer using the LLM.
Invoke the chain with the input dictionary and return the model’s response.
"""
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "Tell me about Nowshad's experience in Huawei?"
llm_chain.invoke({"question": question})

"""
Convert the Chroma vector store into a retriever that performs similarity search.
Build a RetrievalQA chain that feeds retrieved context into the Llama model using the “stuff” method.
The chain automatically fetches relevant document chunks and injects them into the LLM prompt.
Enable verbose mode to display retrieval steps and model reasoning during execution.
Define a natural‑language query asking about Nowshad Ruhani.
Invoke the QA chain to generate an answer based on both the documents and the LLM.
"""
retriever = vectordb.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query = "Tell me about Nowshad Ruhani?"
qa.invoke(query)



##Results from running the above code:
"""
Loading document...
Loaded 1 documents.
Total chunks created: 12
Loading weights: 100%|██████████████████████████████████| 103/103 [00:00<00:00, 6778.49it/s]
Embeddings created and stored in ChromaDB. <langchain_community.vectorstores.chroma.Chroma object at 0x17f0bdd30>
ggml_metal_device_init: tensor API disabled for pre-M5 and pre-A19 devices
ggml_metal_library_init: using embedded metal library
ggml_metal_library_init: loaded in 8.475 sec
ggml_metal_rsets_init: creating a residency set collection (keep_alive = 180 s)
ggml_metal_device_init: GPU name:   MTL0 (Apple M2)
ggml_metal_device_init: GPU family: MTLGPUFamilyApple8  (1008)
ggml_metal_device_init: GPU family: MTLGPUFamilyCommon3 (3003)
ggml_metal_device_init: GPU family: MTLGPUFamilyMetal3  (5001)
ggml_metal_device_init: simdgroup reduction   = true
ggml_metal_device_init: simdgroup matrix mul. = true
ggml_metal_device_init: has unified memory    = true
ggml_metal_device_init: has bfloat            = true
ggml_metal_device_init: has tensor            = false
ggml_metal_device_init: use residency sets    = true
ggml_metal_device_init: use shared buffers    = true
ggml_metal_device_init: recommendedMaxWorkingSetSize  =  5726.63 MB
llama_model_load_from_file_impl: using device MTL0 (Apple M2) (unknown id) - 5460 MiB free
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from /Users/nowshad/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/snapshots/191239b3e26b2882fb562ffccdd1cf0f65402adb/llama-2-7b-chat.Q2_K.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                          general.file_type u32              = 10
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2


LLM initialized.


=> Nowshad experience in Huawei refers to the work history of a person named Nowshad who has worked at Huawei, a multinational technology company. Huawei is a leading provider of telecommunications equipment, smartphones, and other high-tech products. Nowshad's experience at Huawei may involve various roles such as software engineer, product manager, or marketing specialist, among others.
Nowshad's experience at Huawei could include developing and implementing innovative software solutions for Huawei's customers, working on cutting-edge technologies like 5G, artificial intelligence, and the Internet of Things (IoT), and collaborating with cross-functional teams to bring new products to market. Additionally, Nowshad may have

ggml_metal_library_compile_pipeline: compiling pipeline: base = 'kernel_mul_mv_f16_f32_4', name = 'kernel_mul_mv_f16_f32_4_nsg=4'
ggml_metal_library_compile_pipeline: loaded kernel_mul_mv_f16_f32_4_nsg=4                 0x31fa517a0 | th_max = 1024 | th_width =   32

=> worked closely with Huawei's global partners and suppliers to ensure successful project delivery.
Overall, Nowshad's experience in Huawei can provide valuable insights into the company's operations, culture, and business strategies, which could be useful for individuals seeking to work or collaborate with Huawei in the future.

llama_perf_context_print:        load time =   10560.60 ms
llama_perf_context_print: prompt eval time =   10559.75 ms /    90 tokens (  117.33 ms per token,     8.52 tokens per second)
llama_perf_context_print:        eval time =   17124.96 ms /   239 runs   (   71.65 ms per token,    13.96 tokens per second)
llama_perf_context_print:       total time =   28418.06 ms /   329 tokens
llama_perf_context_print:    graphs reused =        237


> Entering new RetrievalQA chain...

Llama.generate: 1 prefix-match hit, remaining 164 prompt tokens to eval

 => Nowshad Ruhani is an AI and software engineer with experience across machine learning (ML), large language models (LLMs), backend systems, and cloud infrastructure. He has worked internationally in Finland, the UK, Australia, and other countries.
 
llama_perf_context_print:        load time =   10560.60 ms
llama_perf_context_print: prompt eval time =    1227.72 ms /   164 tokens (    7.49 ms per token,   133.58 tokens per second)
llama_perf_context_print:        eval time =    2951.58 ms /    55 runs   (   53.67 ms per token,    18.63 tokens per second)
llama_perf_context_print:       total time =    4261.31 ms /   219 tokens
llama_perf_context_print:    graphs reused =         54

> Finished chain.
"""