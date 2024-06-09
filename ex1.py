import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_and_search(input_text, num_results=5):
    # Tokenize the text into words
    words = input_text.lower().split()
    
    # Remove punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Combine words into a single string
    processed_text = ' '.join(words)
    
    # Calculate TF-IDF scores
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([processed_text])
    
    # Get feature names (words) with highest TF-IDF scores
    feature_names = tfidf_vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keyword_indices = scores.argsort()[-5:][::-1]  # Get indices of top 5 keywords
    keywords = [feature_names[idx] for idx in keyword_indices]
    
    # Convert keywords to a single string
    keywords_str = ' '.join(keywords)
    
    # Perform news articles search based on keywords
    
    return keywords_str

r=extract_keywords_and_search("Linux is great")
print(r)
