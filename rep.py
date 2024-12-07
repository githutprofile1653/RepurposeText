import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []



def read_blog(URL):
    r = requests.get(URL)
    soup = BeautifulSoup(r.text, 'html.parser')
    results = soup.find_all(['h1', 'p'])
    text = [result.text for result in results]
    blog = ' '.join(text)
    return blog


def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=110)
    chunks = text_splitter.split_text(text)
    return chunks

def embedding_vector(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_db.save_local("faiss_index")


def chain_prompt_question():

    prompt_template = """
    You are a repurpose AI system, your job is to convert the text or blog into a post in the provided social media platform\n\n
    Context:\n {context}?\n
    social media platform: \n{platform}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "platform"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def get_context_question(input):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    context = new_db.similarity_search(input)

    chain = chain_prompt_question()

    
    response = chain(
        {"input_documents":context, "platform": input}
        , return_only_outputs=True)
    
    return  response["output_text"]
    
    


def main():
    st.set_page_config(page_title="Repurpose AI")

    st.title("Repurpose AI")
    st.subheader("Convert Long form Text content to short Form")
    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>",unsafe_allow_html=True)
    URL= st.text_input("Paste Blog Url and press Enter: ")
    st.write("")
    st.write("")
    if URL:
        
        with st.spinner("Processing..."):
            text = read_blog(URL)
            text_chunks = chunk_text(text)
            embedding_vector(text_chunks)
            
            # st.success("Done")

   
    # submit1 = st.button("Generate Linkdin Post")
    # input = "Linkedin"

    # if submit1:
    #     if URL is not "":
    #         res = get_context_question(input)
    #         st.write("Answer: ", res)
    #         st.session_state['chat_history'].append(("AI", res))
    #         st.session_state['chat_history'].append(("Article", URL))
    #         # Add user query and response to session state chat history
    #         st.write("Read More:",URL)
    #     else:
    #         st.write("")

        

    submit2 = st.button("Generate Twitter Post")
    input = "Twitter: must not be more than 280 characters"

    if submit2:
        if URL is not "":
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
            # Add user query and response to session state chat history
            st.write("Read More:",URL)
        else:
            st.write("")


    submit3 = st.button("Generate Whatsapp Status Post")
    input = "Whatsapp Status: must not be more than 150 characters"

    if submit3:
        if URL is not "":
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
            # Add user query and response to session state chat history
            st.write("Read More:",URL)
        else:
            st.write("")
    submit4 = st.button("Generate Twitter Threads Post")
    input = "Twitter Threads: Each threads must be 150 words"

    if submit4:
       if URL is not "":
            res = get_context_question(input)
            st.write("Answer: ", res)
            st.session_state['chat_history'].append(("AI", res))
            st.session_state['chat_history'].append(("Article", URL))
                # Add user query and response to session state chat history
            st.write("Read More:",URL)
       else:
           st.write("")
    

        # Create sidebar container
    sidebar = st.sidebar
    sidebar.title("History")
    if 'chat_history' in st.session_state:
        for role, text in st.session_state['chat_history']:
            sidebar.write(f"{role}: {text}")

    
        



if __name__ == "__main__":
    main()

    
        





