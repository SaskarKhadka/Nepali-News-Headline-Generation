import streamlit as st
from app.config import tokenizer_config as tok, transformer_config as tfc, attentive_seq2seq_config as seq2seq
from app.config.preprocess import perform_preprocessing

st.set_page_config(
    page_title="Summarization",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def main():
    st.title('Abstractive Summarization of Nepali News Articles')
    st.text_area('Enter Nepali News:', height=300, key='news_input')
    if st.button('Generate Headline'):
        news = st.session_state.news_input
        if news.strip() == '':
            st.error("Please enter some news", icon='🚨')
        
        elif len(news.split()) <= 30:
            st.error("News seems very short, please make sure it has atleast 30 words", icon='🚨')
        
        else:
            text = perform_preprocessing(st.session_state.news_input)
            headline_tformer = tfc.generate(tfc.model, text, tok.tokenizer, tfc.parameters['ENCODER_SEQUENCE_LENGTH'], tfc.parameters['DECODER_SEQUENCE_LENGTH'], tfc.parameters['SOS_TOKEN_ID'], tfc.parameters['EOS_TOKEN_ID'], tfc.parameters['PAD_TOKEN_ID'])
            st.write(f"**Transformer:**&nbsp;&nbsp;&nbsp;{headline_tformer}")
            headline_seq2seq = seq2seq.generate(seq2seq.model, text, tok.tokenizer, seq2seq.parameters['ENCODER_SEQUENCE_LENGTH'], seq2seq.parameters['DECODER_SEQUENCE_LENGTH'], seq2seq.parameters['SOS_TOKEN_ID'], seq2seq.parameters['EOS_TOKEN_ID'], seq2seq.parameters['PAD_TOKEN_ID'])
            st.write("")
            st.write(f"**Attentive Seq2Seq:**&nbsp;&nbsp;&nbsp;{headline_seq2seq}")

if __name__ == "__main__":
    main()
