import os
import streamlit as st
import pandas as pd
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message

# Load Data
def read_data_from_csv(file_path):
    return pd.read_csv(file_path)

folder_path = '/Users/faymajidelhassan/Desktop/Stem-awa/Pages Preprocessed'  # Replace with the actual path to your folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

data_frames = []

for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    data_frame = read_data_from_csv(file_path)
    data_frames.append(data_frame)

df = pd.concat(data_frames, ignore_index=True)

# Ensure the desired columns exist in the DataFrame
desired_columns = ['Keyword','Title', 'Subtitle', 'Summary', 'Search Term', 'Question', 'Answer',"Tags","Sentiment Analysis","Rating"]
for col in desired_columns:
    if col not in df.columns:
        df[col] = None

# Print the columns to verify their names
print(df.columns)

text_columns = ['Keyword','Title', 'Subtitle', 'Summary', 'Search Term', 'Question', 'Answer',"Tags"]

# Fill NaN values with empty strings in text columns
df[text_columns] = df[text_columns].fillna('')

# Create a new column 'combined_text' by joining the text columns
df['combined_text'] = df[text_columns].apply(lambda x: ' '.join(x), axis=1)

# Print the first few rows of the DataFrame to check the result
print(df.head())

# Recommendation functions
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])
user_item_similarity = cosine_similarity(tfidf_matrix)

# Load the 'en_core_web_sm' model only if it's not present
if 'en_core_web_sm' not in spacy.util.get_installed_models():
    spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

def get_content_based_recommendations(query):
    # Transform the user's query using the TF-IDF vectorizer
    query_vector = tfidf_vectorizer.transform([query])
    
    # Compute cosine similarities between the query and all articles
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the top 5 most similar article indices
    similar_articles = cosine_similarities.argsort()[-5:][::-1]
    
    recommendations = df.iloc[similar_articles]
    
    return recommendations

def get_hybrid_recommendations(query, top_n=5):
    # First, get content-based recommendations
    content_rec = get_content_based_recommendations(query)
    
    # If we have a top recommendation from content-based filtering,
    # find other articles similar to it using collaborative filtering
    if not content_rec.empty:
        idx = content_rec.index[0]  # Assuming the first is the most relevant
        sim_scores = list(enumerate(user_item_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get indices of the top-n similar articles
        similar_articles = [i[0] for i in sim_scores[1:top_n+1]]  # Include the first recommendation
        collab_rec = df.iloc[similar_articles]
    else:
        collab_rec = pd.DataFrame()
    
    # Combine the two sets of recommendations
    combined_recommendations = pd.concat([content_rec, collab_rec], ignore_index=True).drop_duplicates()
    
    # Perform sentiment analysis and sort recommendations based on sentiment
    combined_recommendations['Sentiment Analysis'] = combined_recommendations['combined_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    positive_rec = combined_recommendations[combined_recommendations['Sentiment Analysis'] > 0].sort_values(by='Sentiment Analysis', ascending=False)
    neutral_rec = combined_recommendations[combined_recommendations['Sentiment Analysis'] == 0]
    
    sorted_recommendations = pd.concat([positive_rec, neutral_rec], ignore_index=True)
    
    # Sort by sentiment analysis and then by rating
    sorted_recommendations = sorted_recommendations.sort_values(by=['Sentiment Analysis', 'Rating'], ascending=[False, False])

    # Handle the cases where Sentiment Analysis or Rating is missing
    sorted_recommendations = sorted_recommendations.sort_values(by=['Sentiment Analysis', 'Rating'], ascending=[False, False], na_position='last')
    
    return sorted_recommendations[:top_n]  # Limit the number of recommendations to top_n

class ChatHistory:
    
    def __init__(self):
        self.history = st.session_state.get("history", [])
        st.session_state["history"] = self.history

    def default_greeting(self):
        return "Hey 👋, how can I help you ?"

    def initialize_user_history(self):
        if "user" not in st.session_state:
            st.session_state["user"] = []

    def initialize_assistant_history(self):
        if "assistant" not in st.session_state:
            st.session_state["assistant"] = [self.default_greeting()]

    def initialize(self):
        self.initialize_user_history()
        self.initialize_assistant_history()

    def append(self, mode, message):
        st.session_state[mode].append(message)

    def generate_messages(self, container):
        if st.session_state["assistant"]:
            with container:
                for i in range(len(st.session_state["assistant"])):
                    if i < len(st.session_state["user"]):
                        message(
                            st.session_state["user"][i],
                            is_user=True,
                            key=f"history_{i}_user",
                            avatar_style="big-smile",
                        )
                    message(st.session_state["assistant"][i], key=str(i), avatar_style="thumbs")

def chat_interface():
    # Instantiate the ChatHistory class
    history = ChatHistory()
    
    st.title("Hybrid Recommendation Chatbot")

    # Initialize chat history
    history.initialize()

    # Create containers for chat responses and user prompts
    response_container, prompt_container = st.container(), st.container()

    # Display the initial greeting from the chatbot
    with response_container:
        message(st.session_state["assistant"][0], key="initial_greeting", avatar_style="thumbs")

    # Display the prompt form
    user_input = st.text_input("You: ", "")

    # Create a dropdown menu for selecting the number of recommendations
    num_recommendations = st.selectbox("Select number of recommendations", [1, 5, 10, 15, 20], index=1) # Change index to change default number of recommendations

    if user_input:
        if user_input.lower() == 'exit':
            st.write("Goodbye!")
        else:
            recommendations = get_hybrid_recommendations(user_input, top_n=num_recommendations)

            # Determine the type of recommendation message based on user query
            if any(word in user_input.lower() for word in ["what", "where", "how", "when"]):
                response_msg = "Based on your query, here are some recommendations:\n"
            else:
                response_msg = "Here are some suggestions for you:\n"

            if not recommendations.empty:
                links = []
                for idx, row in recommendations.iterrows():
                    if row['Title']:
                        title = row['Title']
                    elif row['Subtitle']:
                        title = row['Subtitle']
                    elif row['Summary']:
                        title = row['Summary']
                    elif row['Question']:
                        title = row['Question']
                    elif row['Tags']:
                        title = row['Tags']
                    else:
                        title = row['Search Term']

                    # title = row['Title'] if row['Title'] else 'Title'
                    # subtitle = row['Subtitle'] if row['Subtitle'] else 'Subtitle'
                    # summary = row['Summary'] if row['Summary'] else 'Summary'
                    # question = row['Question'] if row['Question'] else 'Question'
                    # tags = row['Tags'] if row['Tags'] else 'Tags'
                    # search_term = row['Search Term'] if row['Search Term'] else 'Search Term'

                    # st.write(f"- {title} | {subtitle} | {summary} | {question} | {tags} | {search_term} - ({row['URL']})")

                    links.append(f"{idx +1} - {title}\nLink: [{row['URL']}]({row['URL']})\n")
                
                # Combine all links into a single response message
                response_msg += "\n".join(links)
            else:
                response_msg += "Sorry, I couldn't find any recommendations based on your query."
            
            # Update the chat history
            history.append("user", user_input)
            history.append("assistant", response_msg)
            
            with response_container:
                # Display the chat messages excluding the initial greeting
                for i in range(1, len(st.session_state["assistant"])):
                    message(st.session_state["user"][i-1], is_user=True, key=f"history_{i-1}_user", avatar_style="big-smile")
                    message(st.session_state["assistant"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    chat_interface()

