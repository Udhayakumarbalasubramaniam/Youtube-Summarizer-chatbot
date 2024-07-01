import streamlit as st
import cohere
import numpy as np
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Initialize Cohere client with API key
co = cohere.Client('BYjkWU1SkejRybIuU9nevxt0bKudCJv3AotOVpgC')  # Replace with your actual API key

# Function to extract captions from YouTube video
def extract_captions(video_url):
    video_id = video_url.split("/")[-1].split("?")[0]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# Function to summarize text using LSA
def summarize_text(texts, num_sentences=3):
    if len(texts) == 0:
        return "No captions to summarize."
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    lsa = TruncatedSVD(n_components=1, n_iter=100)
    lsa.fit(X)
    
    terms = vectorizer.get_feature_names_out()
    term_scores = lsa.components_[0]
    
    sentences = sent_tokenize(" ".join(texts))
    sent_scores = np.array([sum([term_scores[vectorizer.vocabulary_.get(term, 0)] for term in sentence.lower().split() if term in vectorizer.vocabulary_]) for sentence in sentences])
    top_sent_indices = sent_scores.argsort()[-num_sentences:][::-1]
    
    summary = " ".join([sentences[i] for i in sorted(top_sent_indices)])
    return summary

# Function to ask LLM (Cohere) for an answer
def ask_llm(question, context, num_generations=1):
    prompt = f"""
    More information about the topic:
    {context}
    Question: {question}
    Extract the answer of the question from the text provided.
    If the text doesn't contain the answer, reply that the answer is not available."""
    
    prediction = co.generate(
        prompt=prompt,
        max_tokens=70,
        model="command-nightly",
        temperature=0.5,
        num_generations=num_generations
    )
    return prediction.generations[0].text


st.title('VidTutor')


if 'conversation' not in st.session_state:
    st.session_state.conversation = []

if 'summary' not in st.session_state:
    st.session_state.summary = ""

video_urls = st.text_input('Enter YouTube video URLs separated by commas:')
question = st.text_input('Enter your question:')

if st.button('Submit'):
    if video_urls and question:
        st.session_state.conversation.append(f"User: {video_urls}, Question: {question}")
        video_urls = video_urls.split(",")
        all_captions = []
        for video_url in video_urls:
            captions = extract_captions(video_url)
            if captions:
                all_captions.extend(captions)

        if all_captions:
            texts = [caption["text"] for caption in all_captions]
            if not st.session_state.summary:
                st.session_state.summary = summarize_text(texts)
            response = ask_llm(question, st.session_state.summary)
            st.session_state.conversation.append(f"Answer: {response}")
            
        
            st.write(f"Answer: {response}")
            st.write(f"Summary: {st.session_state.summary}")
        else:
            error_message = "No captions available or an error occurred."
            st.session_state.conversation.append(f"System: {error_message}")
            st.error(error_message)
    else:
        error_message = "Please provide both video URLs and a question."
        st.session_state.conversation.append(f"System: {error_message}")
        st.error(error_message)

# Display the conversation history
for message in st.session_state.conversation:
    st.write(message)
