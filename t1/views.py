from django.shortcuts import render
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.http import JsonResponse
import networkx as nx
from googletrans import Translator
#from rake_nltk import Rake
import requests
import string
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text as pdf_extract_text
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def input_form(request):
    input_text = request.session.get('input_text', '')
    return render(request, 't1/summarizer.html', {'input_text': input_text})
 

def translator_form(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        target_language = request.POST.get('translate_language')
        
        if input_text and target_language:
            translated_text = translate_text(input_text, target_language)
            return render(request, 't1/translator.html', {'translated_text': translated_text})
        else:
            return JsonResponse({'error': 'Missing input text or target language'}, status=400)
    else:
        return render(request, 't1/translator.html')

def paraphraser_form(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        
        if input_text:
            paraphrased_text = paraphrase_text(input_text)
            return render(request, 't1/paraphraser.html', {'paraphrased_text': paraphrased_text})
        else:
            return JsonResponse({'error': 'Missing input text'}, status=400)
    else:
        return render(request, 't1/paraphraser.html')

def summarize_text(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        # Store the input text in the session
        request.session['input_text'] = input_text
        summary_type = request.POST.get('summary_type')
        translate_language = request.POST.get('translate_language')
        
        # Check if file upload is present
        if request.FILES.get('file'):
            uploaded_file = request.FILES['file']
            input_text += convert_file_to_text(uploaded_file)
        
        # Perform the appropriate summarization based on the summary type
        if summary_type == 'extractive':
            summarized_text = perform_extractive_summarization(input_text)
        elif summary_type == 'abstractive':
            summarized_text = perform_abstractive_summarization(input_text)
        elif summary_type == 'both':
            summarized_text = perform_combined_summarization(input_text)
        else:
            # Handle invalid summary type
            return JsonResponse({'error': 'Invalid summary type'}, status=400)
        
        # Perform translation if translate_language is provided
        translated_text = None
        if translate_language:
            translator = Translator()
            translated_text = translator.translate(summarized_text, dest=translate_language).text
        
        # Perform related news articles search
        related_articles = search_related_articles(input_text)
        
        # Return the summarized text, translated text, and related articles as context to the template
        return render(request, 't1/summarizer.html', {'summarized_text': summarized_text, 'translated_text': translated_text, 'related_articles': related_articles})
    else:
        # Handle invalid request method
        return JsonResponse({'error': 'Invalid request method'}, status=400)


def perform_extractive_summarization(input_text):
    # Tokenize input text into sentences
    sentences = input_text.split('.')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    # Vectorize the sentences using TF-IDF
    count_vectorizer = CountVectorizer(stop_words='english')
    word_count_matrix = count_vectorizer.fit_transform(sentences)
    tfidf_transformer = TfidfTransformer()
    tfidf_matrix = tfidf_transformer.fit_transform(word_count_matrix)
    
    # Calculate similarity matrix using cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Apply TextRank algorithm to get sentence scores
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Sort sentences based on their scores
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    
    # Select top sentences for the summary
    num_sentences = min(5, len(sentences))  # Select top 5 sentences
    top_sentences = ranked_sentences[:num_sentences]
    
    # Join selected sentences to form the summary
    extractive_summary = '. '.join(sentence for _, sentence in top_sentences)
    
    return extractive_summary

def perform_abstractive_summarization(input_text):
    # Load pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

    # Tokenize input text
    inputs = tokenizer(input_text, max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=8, min_length=100, max_length=500, early_stopping=True)
    abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return abstractive_summary

def perform_combined_summarization(input_text, extractive_weight=0.5, abstractive_weight=0.5):
    # Perform extractive summarization
    extractive_summary = perform_extractive_summarization(input_text)
    
    # Perform abstractive summarization
    abstractive_summary = perform_abstractive_summarization(input_text)
    
    # Combine extractive and abstractive summaries
    combined_summary = combine_summaries(extractive_summary, abstractive_summary, extractive_weight, abstractive_weight)
    
    return combined_summary

def combine_summaries(extractive_summary, abstractive_summary, extractive_weight, abstractive_weight):
    """
    Combine extractive and abstractive summaries based on given weights.

    Args:
    - extractive_summary (str): Extractive summary text.
    - abstractive_summary (str): Abstractive summary text.
    - extractive_weight (float): Weight for the extractive summary (0 to 1).
    - abstractive_weight (float): Weight for the abstractive summary (0 to 1).

    Returns:
    - combined_summary (str): Combined summary text.
    """
    # Split summaries into sentences
    extractive_sentences = extractive_summary.split('. ')
    abstractive_sentences = abstractive_summary.split('. ')
    
    # Calculate number of sentences to include from each summary based on weights
    num_extractive_sentences = int(len(extractive_sentences) * extractive_weight)
    num_abstractive_sentences = int(len(abstractive_sentences) * abstractive_weight)
    
    # Select sentences from each summary based on weights
    selected_extractive_sentences = extractive_sentences[:num_extractive_sentences]
    selected_abstractive_sentences = abstractive_sentences[:num_abstractive_sentences]
    
    # Combine selected sentences into final summary
    combined_summary = '. '.join(selected_extractive_sentences) + '. ' + '. '.join(selected_abstractive_sentences)
    
    return combined_summary

def translate_text(input_text, target_language):
    # Initialize the translator
    translator = Translator()
    
    # Translate the input text to the target language
    translated_text = translator.translate(input_text, dest=target_language).text
    
    return translated_text


def search_related_articles(input_text, num_results=4):
    """
    Perform a search for related news articles based on keywords extracted from the input text using TF-IDF.

    Args:
    - input_text (str): The input text to extract keywords from and use for the search.
    - num_results (int): Number of news articles to retrieve (default is 4).

    Returns:
    - related_articles (list of dict): List of related news articles, each containing 'title', 'link', and 'snippet'.
    """
    # Tokenize the input text into words
    words = input_text.lower().split()
    
    # Remove punctuation and stopwords
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
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
    
    # Perform news articles search based on keywords
    related_articles = perform_newsapi_search(' '.join(keywords), num_results=num_results)
    
    return related_articles

def perform_newsapi_search(keywords, num_results=4):
    """
    Perform a search for related news articles based on keywords using News API.

    Args:
    - keywords (str): Keywords to use for the search.
    - num_results (int): Number of news articles to retrieve (default is 4).

    Returns:
    - related_articles (list of dict): List of related news articles, each containing 'title', 'link', 'snippet', and 'image_url'.
    """
    # News API endpoint
    endpoint = "https://newsapi.org/v2/everything"
    
    # Parameters for the search request
    params = {
        "q": keywords,
        "apiKey": "8719081539d34acd9c6cdb1c3417ae48",  # Replace with your actual News API key
        "pageSize": num_results
    }
    
    # Send GET request to the API endpoint
    response = requests.get(endpoint, params=params)
    
    # Check if the request was successful and response contains search results
    if response.status_code == 200:
        # Parse the JSON response
        response_json = response.json()
        
        # Extract articles if available
        if 'articles' in response_json:
            articles = response_json['articles']
            
            # Extract relevant information from articles
            related_articles = []
            for article in articles:
                related_articles.append({
                    "title": article.get('title', ''),
                    "link": article.get('url', ''),
                    "snippet": article.get('description', ''),
                    "image_url": article.get('urlToImage', '')  # Add image URL
                })
            
            return related_articles
        else:
            print("No news articles found.")
            return []
    else:
        print("Error:", response.status_code)
        return []



def convert_file_to_text(file_path):
    """
    Convert a PDF, TXT, or DOCX file to text.

    Args:
    - file_path (str): Path to the input file.

    Returns:
    - text (str): Text extracted from the file.
    """
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as pdf_file:
            text = pdf_extract_text(pdf_file, laparams=dict(all_texts=True), encoding='utf-8')
        return text
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as txt_file:
            text = txt_file.read()
        return text
    elif file_path.endswith('.docx'):
        doc = docx.Document(file_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        text = '\n'.join(paragraphs)
        return text
    else:
        raise ValueError('Unsupported file format. Only PDF, TXT, and DOCX are supported.')


def paraphrase_text(input_text):
    model_name = "tuner007/pegasus_paraphrase"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Split the text into sentences
    sentences = input_text.split(".")
    paraphrases = []

    for sentence in sentences:
        # Clean up sentences

        # remove extra whitespace
        sentence = sentence.strip()

        # filter out empty sentences
        if len(sentence) == 0:
            continue

        # Tokenize the sentence
        inputs = tokenizer.encode_plus(sentence, return_tensors="pt", truncation=True, max_length=512)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # paraphrase
        paraphrase = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            max_length=400,
            early_stopping=True
        )[0]
        paraphrased_text = tokenizer.decode(paraphrase, skip_special_tokens=True)

        paraphrases.append(paraphrased_text)

    # Combine the paraphrases
    combined_paraphrase = " ".join(paraphrases)

    return paraphrased_text



