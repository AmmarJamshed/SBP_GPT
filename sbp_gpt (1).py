# -*- coding: utf-8 -*-
import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import PyPDF2
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Streamlit UI
st.title("State Bank of Pakistan GPT Model")
st.write("This app scrapes PDF links from the SBP website, processes text, and generates responses using GPT.")

# Scrape PDFs from the SBP website
def scrape_sbp_pdfs():
    url = "https://www.sbp.org.pk/l_frame/index2.asp"
    response = requests.get(url)
    web_content = response.content
    soup = BeautifulSoup(web_content, 'html.parser')

    base_url = "https://www.sbp.org.pk"
    pdf_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.endswith('.pdf'):
            if href.startswith('http'):
                full_url = href
            else:
                full_url = base_url + href if href.startswith('/') else base_url + '/' + href
            pdf_links.append(full_url)
    return pdf_links

# Function to download PDFs
def download_pdf(url, directory):
    try:
        local_filename = os.path.join(directory, url.split('/')[-1])
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err} - {url}")
    except Exception as err:
        print(f"Other error occurred: {err} - {url}")

# Download PDFs and display in the UI
pdf_links = scrape_sbp_pdfs()
st.write("PDF Links Found:")
for link in pdf_links:
    st.write(link)

# Option to download PDFs
if st.button("Download PDFs"):
    os.makedirs('pdfs', exist_ok=True)
    for link in pdf_links:
        st.write(f"Downloading {link}")
        download_pdf(link, 'pdfs')
    st.success("PDFs downloaded successfully!")

# Extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Processing extracted PDF text
if st.button("Process PDFs"):
    all_text = ""
    pdf_files = [os.path.join('pdfs', f) for f in os.listdir('pdfs') if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        all_text += extract_text_from_pdf(pdf_file)

    # Save the text for GPT model training or usage
    with open('sbp_text.txt', 'w', encoding='utf-8') as f:
        f.write(all_text)

    st.success("PDF text extracted and saved!")

# GPT-2 Model Loading
@st.cache_resource()
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # Assuming a pre-trained GPT-2 model
    return tokenizer, model

tokenizer, model = load_model()

# Generate text based on a prompt
def generate_text(prompt, model, tokenizer, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )

    # Decode the generated text
    generated_text = [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]
    return generated_text

# Prompt input from the user
prompt = st.text_input("Enter a prompt to generate text:")

# Text generation options
max_length = st.slider("Select max length of generated text", 50, 500, 100)
num_return_sequences = st.slider("Number of generated sequences", 1, 5, 1)

# Button to generate text using GPT-2
if st.button("Generate Text"):
    if prompt:
        st.write("Generating text...")
        generated_text = generate_text(prompt, model, tokenizer, max_length=max_length, num_return_sequences=num_return_sequences)

        for i, text in enumerate(generated_text):
            st.write(f"Generated Text {i+1}:\n{text}\n")
    else:
        st.warning("Please enter a prompt.")
