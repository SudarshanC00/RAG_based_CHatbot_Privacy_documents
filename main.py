
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import SentenceTransformerEmbeddings
from streamlit_chat import message
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from constants import CHROMA_SETTINGS


# checkpoint = "MBZUAI/LaMini-T5-738M"
# print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# base_model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint,
#     device_map=device,
#     torch_dtype=torch.float32
# )



if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory="chroma_db")
    vectorstore.persist()
    print("vectors stored successfully")
    return vectorstore


# @st.cache_resource
def process_answer(vectorstore):
    # qa = qa_llm()
    # pipe = pipeline(
    #     'text-generation',
    #     model = base_model,
    #     tokenizer = tokenizer,
    #     max_length=3000,
    #     truncation=True,
    #     temperature = 0.2
    # )
    # llm = HuggingFacePipeline(pipeline=pipe)
    llm = ChatGoogleGenerativeAI(api_key= 'AIzaSyB6JIb2GyfgAPS4NMd0uZdBsO3DKCUSNnY', model = "gemini-1.5-flash", temperature=0)

    # retriever = vectorstore.as_retriever()

    memory = ConversationBufferMemory(memory_key='chat_history',output_key="answer", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        memory=memory
    )
    return conversation_chain

    # custom_prompt = PromptTemplate(
    #     input_variables=["query"],
    #     template="""
    #     You are an intelligent assistant trained to provide detailed and accurate answers based on the document provided.
    #     Answer the user's question in the following format:
    #     1. Summarize and explain the relevant sections from the document.
    #     2. Highlight the key points and explain their relevance to the query.
    #     3. Provide actionable guidance if applicable.

    #     User's Question: {query}
    #     Your Answer:
    #     """
    # )

    # # Create the final prompt
    # final_prompt = custom_prompt.format(query=instruction["query"])

    # print('final prompt:', final_prompt)

    # generated_text = qa({"query": final_prompt})
    # print('generated_text: ', generated_text)
    # answer = generated_text['result']
    # print('generated answer: ',answer)
    # return answer

def display_conversation(user_input):
    template=f"""
        You are an AI assistant trained to provide short and sweet answers to user questions. Your responses should be:
    
    1. Concise: Use as few words as possible while retaining clarity.
    2. Accurate: Ensure no critical information or key words are missed.
    3. Friendly: Use a polite and approachable tone.

    Always include key terms or phrases relevant to the user's question.

    User Question: {user_input}

    Your Short and Sweet Answer:
    """
    response = st.session_state.conversation({'question': template})
    print(response['answer'])

    st.session_state.chat_history.append({'question':user_input,'answer':response['answer']})

    for i,convo in enumerate(st.session_state.chat_history):
        message(convo['question'],is_user=True,key=str(i) + "_user")
        message(convo['answer'],key=str(i))

    # for i in range(len(history["generated"])):
    #     message(history["past"][i], is_user=True, key=str(i) + "_user")
    #     message(history["generated"][i],key=str(i))

# def handle_userinput(user_question,conversation_chain):
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     st.session_state.chat_history.append({"role":"user","content":user_question})
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     with st.chat_message("assistant"):
#         response = conversation_chain({"question": user_question})
#         assistant_response = response["answer"]
#         st.markdown(assistant_response)
#         st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})


# Function to read a local PDF file and extract its text
def read_pdf(file_path):
    try:
        # Create a PDF reader object
        reader = PdfReader(file_path)
        
        # Extract text from each page
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text() + "\n"
        
        return pdf_text
    except Exception as e:
        return f"Error reading PDF: {e}"


def main():
    load_dotenv()
    st.set_page_config(page_title="OpenAI Chatbot",
                       page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    st.header("Chatbot for all your queries about services at OpenAI 🤖")

    file_path = "/teamspace/studios/this_studio/OPENAI Privacy Document.pdf" 
    pdf_content = read_pdf(file_path)

    # get the text chunks
    text_chunks = get_text_chunks(pdf_content)

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # Initialize session state for generated responses and past messages
    # if "generated" not in st.session_state:
    #     st.session_state["generated"] = ["I am ready to help you"]
    # if "past" not in st.session_state:
    #     st.session_state["past"] = ["Hey there!"]
    st.session_state.conversation = process_answer(vectorstore)        
    user_input = st.chat_input("Ask AI....")
    if user_input:
            display_conversation(user_input)
            # Search the database for a response based on user input and update session state
    

if __name__ == '__main__':
    main()

