import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline

from googletrans import Translator
translator = Translator()

class TextGenerator:
    def __init__(self):
        self.generator: TextGenerationPipeline
        self.max_length = 300
        set_seed(1)

    def load_generator(self) -> None:  
        self.generator = pipeline('text-generation', model='gpt2')
      
    def generate_text(self, starting_text: str) -> str:
        return self.generator(starting_text,
                              max_length=self.max_length,
                              num_return_sequences=1)[0]['generated_text']

@st.cache(allow_output_mutation=True)
def instantiate_generator():
    generator = TextGenerator()
    generator.load_generator()
    return generator

if __name__ == '__main__':
    st.title('GPT-2 em português')

    text_unlim = st.text_area("Area for text", "Text to Display")
    generator = translator.translate(starting_text, src= 'pt',dest='en')							
    generator = instantiate_generator()

    if starting_text:
        response = generator.generate_text(starting_text)
        result = translator.translate(response, src= 'en',dest='pt')
        st.markdown(f'Completed phrase: {result}')

