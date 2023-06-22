import numpy as np
import pandas as pd
import pickle
import json
import random
import re
import sys
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Download once
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

pd.set_option('mode.chained_assignment', None)

# Dataset (preprocessing can be found in preprocessing.ipynb)
# =======================================
raw_data = pd.read_csv('data/cleaned_anime.csv')
for column in ['genres', 'themes', 'producers', 'licensors', 'demographics', 'studios']:
    raw_data[column] = raw_data[column].fillna("")

raw_data['cleaned_title'] = raw_data['cleaned_title'].astype(str)
raw_data['cleaned_title_english'] = raw_data['cleaned_title_english'].astype(str)
# =======================================
    
# Global variables
# =======================================
# Spacy NLP model
lemmatizer = WordNetLemmatizer()

responses = {
    'introduce': [
        "<b><u>Welcome anime enthusiasts!</u></b><br>I'm <b>RecAnime</b>, your friendly anime recommender chatbot. Looking for your next binge-worthy anime? Just tell me your preferences, and I'll suggest some fantastic shows that match your taste. Whether you're into action, romance, or mystery, I've got a recommendation for you!<br><i>For more information, you can type <b>Help me</b> or <b>Guide me</b> to get started with my features!</i>"
    ],
    'greetings': [
        'Hey there! How can I help you today :)?',
        'Hello, nice to meet you!',
        'Hello, How can I help you?',
        'Hi there! Need some anime recommendations? Just let me know!',
        "Hello! Looking for some anime? I'm here to assist!",
        "Greetings! How can I assist you today?"
    ],
    'help_me': [
        "Sure thing! I'm here to assist you in finding the perfect anime. Here are some features that you can use:<br>- <b>Recommend Anime</b>: Tell me a specific anime that you enjoyed, and I'll provide the recommendations just for you! If you don't have any, I can still suggest you some anime but I need to ask some questions to you :D<br>Example: Can you recommend me anime?, I'm interested with Kimi No Na Wa, suggest me something similar!<br><br>- <b>Search Anime</b>: Search for a specific anime by its title so you can get more information about it<br>Example: Can you search \"Attack on Titan\" for me<br><i>Make sure to wrap the query with <b>quotes</b>!</i>"
    ],
    'anime_recommended': [
        'Since you enjoyed <b>{}</b>, I think you might also like these anime!',
        'Since you liked <b>{}</b>, I have some great recommendations for you!',
        'If you enjoyed <b>{}</b>, I believe you might enjoy these anime as well!'
    ],
    'recommended':[
        'Based on your preference, I recommend checking out these anime!',
        "I've analyzed your preference, and I think you might enjoy these anime!",
        "After considering your interests, I suggest these anime for you"
    ],
    'no_recommend':[
        'I apologize, I could not find any anime that suits with your preference :(',
        "I'm sorry, but I could not generate any recommendations based on your preference :("
    ],
    'ask_anime_pref': [
        "Before I can give you recommendations, I'd like to know more about your anime preferences. Do you have a favorite anime that you liked? Let me know, and I'll find similar ones that you might enjoy!<br><i>Example: <b>I enjoyed Kimi No Na Wa movie</b></i><br><i>You can type <b>skip</b> to skip this question</i>"
    ],
    'anime_not_found': [
        'I apologize, looks like I could not find the anime in my database. You can try to search the anime first to check.',
        "I'm sorry, but I could not find the anime in my database. You can try to search the anime first to check"
    ],
    'ask_genres':[
        'Alright! Can you share me what kind of genres do you enjoyed the most?<br><i>Please separate each gender with commas, example: <b>Action, Drama, Fantasy</b></i><br><i>You can type <b>skip</b> to skip this question</i>'
    ],
    'genres_not_found': [
        'Sorry, I could not find any genres that suits with my database. You can check <a target="_blank" href="https://myanimelist.net/anime.php">here</a> to see the list of genres...'
    ],
    'ask_type': [
        'Right! Can you share me What kind of anime types do you want?<br><i>Please just specify one type, example: <b>Movie</b></i><br><i>You can type <b>skip</b> to skip this question</i>'
    ],
    'type_not_found': [
        "Sorry, I could not find that type in my database. Note that there are only <i>TV, Movie, Special, OVA, and ONA</i>"
    ],
    'ask_desc': [
        'Okay! Are there any specific description that you wanted?<br><i>Example: <b>I liked to watch anime about volleyball and team competition!</b></i><br><i>You can type <b>skip</b> to skip this question</i>'
    ],
    'search_found':[
        'Searching for <b>"{}"</b>....<br>Here are anime that I found!',
        'Searching for <b>"{}"</b>....<br>Looks like I found some!'
    ],
    'search_not_found':[
        'Sorry, it seems I cannot find "{}" in my database :(',
        'Oops! I think I could not find "{}" in my database :('
    ],
    'search_query_not_found':[
        "Hmmm, it seems you did not specify the query that you want to search :/",
        "Oops! it seems like you fogot to provide a search query.",
        "Sorry.. it seems you forget to specify a search query. If you are confused, feel free to type <b>Help</b>!"
    ],
    'thanks': [
        'Happy to help!', 'No problem at all!', 'Any time!', 'You are welcome!', 'My pleasure'
    ],
    'reset': [
        'You have reset the chatbot to the initial state! Feel free to ask me for anime recommendations or type <b>help</b> if you need assistance.'
    ],
    'no_intent': [
        "I'm sorry, but I could not comprehend your input. If you are confused, you can type <b>help</b> to check my features!",
        "I'm apologize, but I'm having a difficulty to understand your request...",
        "Apologies, but I'm unable to understand what you're asking..."
    ]
} 

unique_genres = [
    'Action', 'Comedy', 'Drama', 'Fantasy','Mystery', 'Suspense', 'Avant Garde',   
    'Adventure', 'Award Winning', 'Supernatural', 'Sci-Fi',  'Romance', 'Slice of Life', 
    'Gourmet', 'Horror', 'Sports', 'Erotica','Ecchi', 'Boys Love', 'Girls Love', 'Hentai',
]
unique_types = ['TV', 'Movie', 'OVA', 'ONA', 'Special']

# Regex title matching (original & english titles)
regex_title_pattern = '|'.join([r"\b" + re.escape(title) + r"\b" for title in sorted(raw_data['cleaned_title'], key=lambda x: (-len(x), x)) if title != ""])
regex_title_english_pattern = '|'.join([r"\b" + re.escape(title) + r"\b" for title in sorted(raw_data['cleaned_title_english'], key=lambda x: (-len(x), x)) if title != ""])
# =======================================

# Chatbot model
class Chatbot:
    def __init__(self, model):
        self.chats = []
        self.model = load_model(model) # load the pretrained model
        self.error_threshold = 0.75 # threshold for the intent prediction
        
        self.intents = json.loads(open("data/intents.json").read())
        self.words = pickle.load(open("data/words.pkl", "rb"))
        self.labels = pickle.load(open("data/labels.pkl", "rb"))
        
        # user preference data
        self.expected_intent = ''
        self.user_genres = []
        self.user_type = ''
        self.user_pref = ''
        
        # Initial chat
        self.add_chat("bot", random.choice(responses['introduce']))
        
    def add_chat(self, who, message, items=[]):
        currentTime = datetime.now().strftime("%I:%M %p")
        response = {"who": who, "message": message, "time": currentTime, "items": items}
        self.chats.append(response)
        return response
    
    def reset(self):
        # Reset bot to initial state
        self.chats = []
        self.user_genres = []
        self.user_type = ''
        self.user_pref = ''
        self.expected_intent = ''
        
        self.add_chat("bot", random.choice(responses['reset']))
        
    def tokenize_and_lemmatize(self, query):
        sentence_words = nltk.word_tokenize(query)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bow(self, sentences):
        # Bag of words
        bag = [0] * len(self.words)
        for sentence in sentences:
            for idx, w in enumerate(self.words):
                if w == sentence:
                    bag[idx] = 1
        return np.array(bag)
    
    def predict_intent(self, bag_of_words):
        res = self.model.predict(np.array([bag_of_words]))[0] # type: ignore
        results = [[i, r] for i, r in enumerate(res) if r > self.error_threshold]
        results.sort(key=lambda x: x[1], reverse=True)
        if not results:
            return 'no_intent'
        return self.labels[results[0][0]]
    
    def convert_data_to_list(self, data):
        data = data[['title', 'title_english', 'type', 'synopsis', 'main_picture', 'rating', 'score', 'scored_by', 'genres', 'url']]
        return data.to_dict('records')

    def user_recommendation(self, data, anime_id=-1):
        anime_indexes = []
        # If user specify it's anime preference
        if anime_id != -1:
            anime_id_list = pd.Series(data.index, index=data['mal_id'])
            idx = anime_id_list[anime_id]
            
            tfidf_vect = TfidfVectorizer(analyzer='word',ngram_range=(1, 2), min_df=0, stop_words='english')
            tfidf_matrix = tfidf_vect.fit_transform(data['description'])
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            # # Use the preprocessed cosine similarity matrix
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Take top 30 anime except the user preference one
            sim_scores = sim_scores[1:31]
            anime_indexes = [i[0] for i in sim_scores]
        else:
            anime_docs = data['description'].tolist()
            anime_docs.append(self.user_pref)
            # TF-IDF
            tfidf_vect = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
            tfidf_vect.fit_transform(anime_docs)
            # Cosine similarity
            cosine_sim = cosine_similarity(tfidf_vect.transform([self.user_pref]), tfidf_vect.transform(anime_docs[:-1]))
            sim_scores = [(i, score) for i, score in enumerate(cosine_sim[0])]
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            # Take top 30 anime
            sim_scores = sim_scores[0:31]
            anime_indexes = [i[0] for i in sim_scores]

        candidates = data.iloc[anime_indexes].copy()
        candidates = self.weighted_rating(candidates)
        # Only take top 5 candidates
        candidates = candidates.head(5)
        return self.convert_data_to_list(candidates)
    
    def weighted_rating(self, data):
        list_members = data[data['members'].notna()]['members'].astype('int')
        list_scores = data[data['score'].notna()]['score'].astype('float')
        
        avg_score = list_scores.mean()
        quant_members = list_members.quantile(0.60)
        # only take anime that has more than quantile of the total members
        qualified = data[(data['members'] >= quant_members) & (data['members'].notnull()) & (data['score'].notnull())]
        qualified['members'] = qualified['members'].astype('int')
        qualified['score'] = qualified['score'].astype('float')
        
        def sort_by_weighted_rating(x, m, C):
            v = x['members']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        qualified['weight_rating'] = qualified.apply(lambda x: sort_by_weighted_rating(x, quant_members, avg_score), axis=1)
        qualified = qualified.sort_values('weight_rating', ascending=False)
        return qualified

    def search(self, query):
        # Search by original title
        result_search = raw_data[raw_data['cleaned_title'] == query]
        # Search by english title
        result_search = pd.concat([result_search, raw_data[raw_data['cleaned_title_english'] == query]], ignore_index=True)
        # Drop duplicate
        result_search = result_search.drop_duplicates(keep='first')
        if result_search.shape[0] < 5:
            # Search by original title (substring)
            result_search = pd.concat([result_search, raw_data[raw_data['cleaned_title'].str.contains(query)]], ignore_index=True)
            # Drop duplicate
            result_search = result_search.drop_duplicates(keep='first')
            if result_search.shape[0] < 5:
                # Search by english title (substring)
                result_search = pd.concat([result_search, raw_data[raw_data['cleaned_title_english'].str.contains(query)]], ignore_index=True)
                # Drop duplicate
                result_search = result_search.drop_duplicates(keep='first')
        # Get 5 top result
        result_search = result_search.head(5)
        return self.convert_data_to_list(result_search)
    
    def chat(self, query):
        current_intent = self.expected_intent
        cleaned_query = cleaning_text(query)
        
        items = []
        response = ""
        
        # Getting user intent if the current intent is not specified
        if not current_intent:
            tokens = self.tokenize_and_lemmatize(cleaned_query)
            bow = self.bow(tokens)
            current_intent = self.predict_intent(bow)
            print('intent:', current_intent)
        
        # Intent selection
        if current_intent == 'recommend_anime':
            anime_title = extract_anime(cleaned_query)
            
            if anime_title:
                # if the user specify the anime preference
                anime_id = get_anime_id(anime_title)
                current_genres = raw_data[raw_data['mal_id'] == anime_id]['genres'].values[0].split(',')
                filtered_data = filter_by_genre(raw_data, current_genres)
                filtered_data = filtered_data.reset_index(drop=True)

                items = self.user_recommendation(filtered_data, anime_id=anime_id)
                
                if not items:
                    response = random.choice(responses['no_recommend'])
                else:
                    response = random.choice(responses['anime_recommended']).format(str(anime_title).title())
                
                self.expected_intent = ''
            else:
                # else we need to ask some questions
                response = random.choice(responses['ask_anime_pref'])
                self.expected_intent = 'ask_anime_pref'
        
        elif current_intent == 'search':
            search_query = extract_quotes(query)
            cleaned_search_query = cleaning_text(search_query)
            
            if search_query:
                items = self.search(cleaned_search_query)
                # if no anime found
                if not items:
                    response = random.choice(responses['search_not_found']).format(str(search_query))
                else:
                    response = random.choice(responses['search_found']).format(str(search_query))
            else:
                # if there is no query
                response = random.choice(responses['search_query_not_found'])
            
            self.expected_intent = ''
                
        elif current_intent == 'ask_anime_pref':
            if query.lower().strip() == 'skip':
                response = random.choice(responses['ask_genres'])
                self.expected_intent = 'ask_genres'
            else:
                anime_title = extract_anime(cleaned_query)
                if anime_title:
                    anime_id = get_anime_id(anime_title)
                    current_genres = raw_data[raw_data['mal_id'] == anime_id]['genres'].values[0].split(',')
                    filtered_data = filter_by_genre(raw_data, current_genres)
                    filtered_data = filtered_data.reset_index(drop=True)
                    items = self.user_recommendation(filtered_data, anime_id=anime_id)
                    if not items:
                        response = random.choice(responses['no_recommend'])
                    else:
                        response = random.choice(responses['anime_recommended']).format(str(anime_title).title())
                    
                    self.expected_intent = ''
                else:
                    response = random.choice(responses['anime_not_found'])
                    self.expected_intent = 'ask_anime_pref'
        
        elif current_intent == 'ask_genres':
            self.user_genres = []
            if query.lower().strip() == 'skip':
                response = random.choice(responses['ask_type'])
                self.expected_intent = 'ask_type'
            else:
                current_genres = extract_by_commas(query)
                for current_genre in current_genres:
                    if current_genre in [genre.lower() for genre in unique_genres]:
                        self.user_genres.append(current_genre)

                if self.user_genres:
                    response = random.choice(responses['ask_type'])
                    self.expected_intent = 'ask_type'
                else:
                    # if there is no genre found, then ask again
                    response = random.choice(responses['genres_not_found'])
                    self.expected_intent = 'ask_genres'
                    
        elif current_intent == 'ask_type':
            self.user_type = ''
            if query.lower().strip() == 'skip':
                response = random.choice(responses['ask_desc'])
                self.expected_intent = 'ask_desc'
            else:
                current_type = cleaned_query
                if current_type in [type.lower() for type in unique_types]:
                    self.user_type = current_type
                
                if self.user_type:
                    response = random.choice(responses['ask_desc'])
                    self.expected_intent = 'ask_desc'
                else:
                    # if there is no anime type found, then ask again
                    response = random.choice(responses['type_not_found'])
                    self.expected_intent = 'ask_type'

        elif current_intent == 'ask_desc':
            # extract noun phrases and create as document
            self.user_pref = ''
            if query.lower().strip() != 'skip':
                noun_phrases = extract_noun(query)
                self.user_pref = ' '.join(noun_phrases)
            
            filtered_anime = filter_by_genre(raw_data, self.user_genres)
            filtered_anime = filter_by_type(filtered_anime, self.user_type)
            if filtered_anime.empty:
                response = random.choice(responses['no_recommend'])
            else:
                items = self.user_recommendation(filtered_anime, anime_id=-1)
                if not items:
                    response = random.choice(responses['no_recommend'])
                else:
                    response = random.choice(responses['recommended'])
            
            self.expected_intent = ''
            
        
        elif current_intent:
            response = random.choice(responses[current_intent])
            
        return response, items

# Other utilities function
# ========================
def get_anime_id(query):
    result_id_english = raw_data[raw_data['cleaned_title_english'] == query]
    result_id = raw_data[raw_data['cleaned_title'] == query]
    
    if not result_id_english.empty:
        return result_id_english['mal_id'].iloc[:1].values[0]
    elif not result_id.empty:
        return result_id['mal_id'].iloc[:1].values[0]
    return -1
    
def extract_quotes(query):  
    # Extract anime
    regex_quotes = r'([\'"])(.*?)\1'
    matches_quotes = re.findall(regex_quotes, query)
    extracted = [match[1] for match in matches_quotes]
    
    if len(extracted) == 0:
        return ""
    
    return extracted[0]

def extract_anime(query):
    english_titles = re.findall(regex_title_english_pattern, query, flags=re.IGNORECASE)
    titles = re.findall(regex_title_pattern, query, flags=re.IGNORECASE)
    if len(english_titles) > 0:
        return english_titles[0]
    elif len(titles) > 0:
        return titles[0]
    return ""
    
def extract_by_commas(text):
    # text: expecting each value should be separated with commas (,)
    informations = text.split(',')
    for idx, information in enumerate(informations):
        information = re.sub(' +', ' ', information)
        information = information.strip().lower()
        informations[idx] = information
    return informations

def extract_noun(text):
    # Extracting noun phrases....
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    noun_phrases = [word for word, pos in tagged_words if pos.startswith('NN')]
    return noun_phrases
        
def cleaning_text(text):
    text = str(text)
    text = re.sub("-",' ', text)
    # Removing punctuations
    text = re.sub(r'[^\w\s]', ' ', text)
    # Removing unicode characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Removing continous spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading spaces
    text = text.strip()
    # lowercase
    text = text.lower()
    return text
    
def removing_punctuation(text):
    clean = re.sub(r"""[,.;@#?!&$]+\ *"""," ", text, flags=re.VERBOSE)
    return clean.strip()
    
def filter_by_genre(data, genres):
    if genres:
        return data[data['genres'].apply(lambda x: any(genre.lower() in x.lower().strip(',') for genre in genres))]
    return data

def filter_by_type(data, anime_type):
    return data[data['type'].str.contains(anime_type, flags=re.IGNORECASE)]
    
# If you want to test, you can run this file
# ==========================================
if __name__ == "__main__":
    chatbot = Chatbot("chatbot_model.h5")
    
    while True:
        user_input = input("You: ")
        response, items = chatbot.chat(user_input)
        print("Bot:", response)
        for idx, item in enumerate(items):
            print("{}. {} [score: {}]".format(idx+1, item['title'], item['score']))