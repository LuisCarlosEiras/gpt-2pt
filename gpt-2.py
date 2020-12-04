import warnings
warnings.filterwarnings("ignore")

import gc
gc.collect()

import streamlit as st
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline

from google_trans_new import google_translator 
translator = google_translator()

translate_text = translator.translate(p.text, lang_tgt='pt')  
st.write(translate_text)
st.write("""*Fonte*: Netcraft""")

class TextGenerator:
    def __init__(self):
        self.generator: TextGenerationPipeline
        self.max_length = 50
        set_seed(1)

    def load_generator(self) -> None:  
        self.generator = pipeline('text-generation', model='gpt2')
      
    def generate_text(self, text_unlim: str) -> str:
        return self.generator(text_unlim,
                              max_length=self.max_length,
                              num_return_sequences=1)[0]['generated_text']

@st.cache(allow_output_mutation=True)
def instantiate_generator():
    generator = TextGenerator()
    generator.load_generator()
    return generator

if __name__ == '__main__':
    st.title('GPT-2 em português')

    text_unlim = st.text_area("Escreva suas palavras ou frases abaixo", "Escreva aqui e clique Ctrl + Enter")
    translate_text = translator.translate(text_unlim,  lang_tgt='pt')  
    generator = translate_text()
     
    if text_unlim:
        response = generator.generate_text(text_unlim)
        translator = Translator(to_lang = "pt")
        result = translator.translate(response)                
        st.markdown(f'Completed phrase: {result}')            

    
        

