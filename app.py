import tensorflow_hub as hub
import re
import gradio as gr
import urllib.request
import numpy as np
import openai
import os
from sklearn.neighbors import NearestNeighbors

MODEL_NAME = "gpt-3.5-turbo"
BEFORE_TEXT = 1
AFTER_TEXT = 5
MAX_TOKENS=2000
CHUNKS_LENGTH = 150

def preprocess(text_list):
    # Replace a substring in each string and remove extra spaces
    processed_list = [re.sub('\s+', ' ', text.replace('\n', ' ')) for text in text_list]
    return processed_list

def get_text(path):
    # Open the text file and read its contents into a list
    with open(path, 'r') as file:
        text_list = [line for line in file]
    
    # Preprocess the text_list and return the result
    return preprocess(text_list)


def text_to_chunks(text_list, chunk_length=CHUNKS_LENGTH):
    chunks_list = []

    text = ' '.join(text_list)
    words = text.split(' ')
    num_chunks = len(words) // chunk_length + int(len(words) % chunk_length > \
0)

    for i in range(num_chunks):
        start = i * chunk_length
        end = min((i + 1) * chunk_length, len(words))
        chunk = ' '.join(words[start:end]).strip()
        formatted_chunk = f'[{i}] "{chunk}"'
        chunks_list.append(formatted_chunk)

    return chunks_list

def load_searcher(path, chunks_length=CHUNKS_LENGTH):
    global searcher

    text_list = get_text(path)
    chunks_list = text_to_chunks(text_list, chunks_length)
    searcher.train(chunks_list)
    return 'Corpus Loaded.'

class TextSimilarity:
    """
    A text similarity search using the Universal Sentence Encoder (USE) from TensorFlow Hub
    to find the most semantically related texts within a collection.
    """

    def __init__(self):
        """Initializes the TextSimilarity class and loads the USE model."""
        self.use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.is_fitted = False

    def train(self, dataset, batch_size=1000, num_neighbors=10):
        """
        Trains the Nearest Neighbors model on the embeddings of the dataset.

        :param dataset: List of texts to train the model.
        :param batch_size: Size of batches for processing texts.
        :param num_neighbors: Number of nearest neighbors to consider.
        """
        self.dataset = dataset
        self.embeddings = self.generate_embeddings(dataset, batch_size=batch_size)
        num_neighbors = min(num_neighbors, len(self.embeddings))
        num_neighbors = max(num_neighbors, 5)  # Ensure at least 5 neighbors are returned, change the 5 to any minimum neighbors you want

        self.neighbor_model = NearestNeighbors(n_neighbors=num_neighbors)
        self.neighbor_model.fit(self.embeddings)
        self.is_fitted = True

    def __call__(self, query, before=BEFORE_TEXT, after=AFTER_TEXT, return_texts=True):
        """
        Retrieves the nearest neighbors of the input query in the dataset.

        :param query: Input text to search for related texts.
        :param return_texts: Whether to return the actual texts or their indices.
        :return: List of related texts or their indices.
        """
        if not self.is_fitted:
            raise ValueError("The model is not trained. Call the 'train' method first.")
        
        query_emb = self.use_model([query])
        nearest_neighbors = self.neighbor_model.kneighbors(query_emb, return_distance=False)[0]

        if return_texts:
            # returns a range of elements from the dataset that includes the 5
            # elements before and the 10 elements after each index in
            # nearest_neighbors,
            before = before
            after = after

            return ['\n\n'.join(self.dataset[max(0, i-before):i+after]) if (i-before)-(i+after) > 1 else self.dataset[i] for i in nearest_neighbors]

            #return [' '.join(self.dataset[max(0, i-before):i+after]) for i in nearest_neighbors]
        else:
            return nearest_neighbors

    def generate_embeddings(self, input_texts, batch_size=1000):
        """
        Creates embeddings for a list of input_texts using the USE model.

        :param input_texts: List of texts to create embeddings for.
        :param batch_size: Size of batches for processing texts.
        :return: Numpy array of text embeddings.
        """
        all_embeddings = []
        for i in range(0, len(input_texts), batch_size):
            texts_batch = input_texts[i:(i+batch_size)]
            emb_batch = self.use_model(texts_batch)
            all_embeddings.append(emb_batch)
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings


def download_text(url, output_path):
    """
    Downloads a text file from a URL and saves it to the specified output path.

    :param url: URL of the text file to download.
    :param output_path: Path where the downloaded file should be saved.
    """

    # Check if the output directory exists, and create it if necessary
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    urllib.request.urlretrieve(url, output_path)


def generate_text(openAI_key, prompt, model=MODEL_NAME):
    """
    Generates text using OpenAI's GPT-3.5-turbo model (or another specified model).

    :param openAI_key: The API key for accessing OpenAI's GPT-3.5-turbo model.
    :param prompt: The text prompt to generate a response for.
    :param model: The name of the OpenAI model to use (default: "gpt-3.5-turbo").
    :return: The generated response from the assistant.
    """
    try:
        openai.api_key = openAI_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant writing an article."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Extract the assistant's response from the returned object
        assistant_response = response.choices[0].message['content']
        return assistant_response
    except Exception as e:
        return(f"Error generating text: {e}")

def truncate_tokens(text, max_tokens):
    tokens = text.split()
    truncated_tokens = tokens[:max_tokens]
    truncated_text = ' '.join(truncated_tokens)
    return truncated_text



def generate_answer(query, openAI_key, before=BEFORE_TEXT, after=AFTER_TEXT, max_tokens=MAX_TOKENS):
    top_chunks = searcher(query, before, after)

    print(">>>> top chunks")
    print(top_chunks)
    print("<<<<")
    print("*************************")
    #search_results  = ""
    #for c in top_chunks:
    #search_results = '\n\n'.join(top_chunks)
    search_results = '\n\n\n'.join(top_chunks) if len(top_chunks) > 1 else top_chunks


    
    search_results = truncate_tokens (search_results, max_tokens)
    
    prompt = f"query: {query}\n" \
       "Please compose a reply to the query above using only the "\
       "information provided in the following search results. " \
       "If the search results do not relate to the query, please output " \
       "'Text not found'\n" \
       "search results: \n\n" \
       f"{search_results}"
    
    print(">>>>**** prompt")
    print(prompt)
    print("<<<<****")

    answer = generate_text(openAI_key, prompt, MODEL_NAME) 
    return answer, prompt
    
def question_answer(url, file, query, max_tokens, before_text, after_text, chunks_length, openAI_key):
    if openAI_key.strip()=='':
        return '[ERROR]: Please enter you Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'
    if url.strip() == '' and file == None:
        return '[ERROR]: Both URL and Text File is empty. Provide atleast one.'
    
    if url.strip() != '' and file != None:
        return '[ERROR]: Both URL and Text File is provided. Please provide only one (eiter URL or Text File).'

    chunks_length_number = get_integer_from_textbox(chunks_length, CHUNKS_LENGTH)
    
    if url.strip() != '':
        glob_url = url
        download_text(glob_url, 'corpus.txt')
        load_searcher('corpus.txt', chunks_length_number)

    else:
        old_file_name = file.name
        file_name = file.name
        file_name = file_name[:-12] + file_name[-4:]
        os.rename(old_file_name, file_name)

        load_searcher(file_name, chunks_length_number)

    if query.strip() == '':
        return '[ERROR]: Question field is empty'

    before_number = get_integer_from_textbox(before_text, BEFORE_TEXT)
    after_number = get_integer_from_textbox(after_text, AFTER_TEXT)
    max_tokens = get_integer_from_textbox(max_tokens, MAX_TOKENS)
    return generate_answer(query,openAI_key, before_number, after_number, max_tokens)

def get_integer_from_textbox(input_text, default=1):
    if input_text.strip():  # Check if the input text is not empty or whitespace
        try:
            value = int(input_text)
            return value
        except ValueError:  # If the input text cannot be converted to an integer
            return default
    else:
        return default


searcher = TextSimilarity()

title = 'Apantas'
description = """ Apantas allows you to ask questions from a text file using Universal Sentence Encoder and Open AI."""

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        
        with gr.Group():
            gr.Markdown(f'<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>')
            openAI_key=gr.Textbox(label='Enter your OpenAI API key here')
            url = gr.Textbox(label='Enter Text File URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            f = gr.File(label='Upload your Text file here', file_types=['.txt'])
            before_text = gr.Textbox(label="Search parameter - before - default 1" )
            after_text = gr.Textbox(label="Search parameter - after - default 5")
            max_tokens = gr.Textbox(label="Max Tokens - default 2000")
            chunks_length = gr.Textbox(label="Chunks Length - default 150")
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :').style(show_copy_button=True)
            thinking = gr.Textbox(label="The input to OpenAI").style(show_copy_button=True)
            
        btn.click(question_answer, inputs=[url, f, question, max_tokens, before_text, after_text, chunks_length, openAI_key], outputs=[answer, thinking])
#openai.api_key = os.getenv('Your_Key_Here') 

demo.launch()
