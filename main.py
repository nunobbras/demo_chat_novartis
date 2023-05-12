from PIL import Image
import streamlit as st

from langchain.llms import OpenAI

import os
from llama_index import GPTSimpleVectorIndex, download_loader, LLMPredictor, PromptHelper, ServiceContext

from streamlit_chat import message

OPENAI_API_KEY = "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" #@param {type:"string"}
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def rebuild_model():
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=temperature, model_name="text-davinci-003"))
    # define prompt helper
    # set maximum input size
    max_input_size = 4076
    # set number of output tokens
    num_output = 512
    # set maximum chunk overlap
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    global index
    index = GPTSimpleVectorIndex.load_from_disk('index.json', service_context=service_context)

with st.sidebar:
    temperature = st.slider("Temperature",
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        on_change=rebuild_model,
        step=0.1
    )
    answer_in_english = st.checkbox('Answer in English')
    indicate_sources = st.checkbox('Show sources', value=True)
    st.title("Topics covered")
    st.text(
    """
            - Research Disease Areas
                - Cardiovascular and metabolic disease research at Novartis
                - Diseases of Aging and Regenerative Medicine
                - Exploratory Disease Research (DAx)
                - Immunology
                - Neuroscience
                - Oncology
                - Ophthalmology
                - Tropical Diseases
    """)

rebuild_model()

image = Image.open('./logo_novartis.png')


with st.columns(3)[1]:
    st.image(image, width=200)


def generate_response(prompt):
    if answer_in_english:
        prompt += 'Answer in Engish\n'
    else:
        prompt += 'Tens de responder em Português Europeu\n'

    return index.query(
        prompt,
        mode="default",
        response_mode="default"
    )


# #Creating the chatbot interface
# st.title("NOS Chat GPT Demo")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    response=output.response.strip()
    if indicate_sources:
        response += f'\n \n \n References:'
        for source in output.source_nodes:
            response += f'\n- {source.extra_info["source"]}'
            response += f'\n \n Similarity: {round(source.similarity * 100)}'
    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["generated"][i], key=str(i))