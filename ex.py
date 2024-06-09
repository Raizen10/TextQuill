import requests

def perform_newsapi_search(keywords,num_results=4):
    """
    Perform a search for related news articles based on keywords using News API.

    Args:
    - keywords (str): Keywords to use for the search.
    - api_key (str): Your News API key.
    - num_results (int): Number of news articles to retrieve (default is 4).

    Returns:
    - related_articles (list of dict): List of related news articles, each containing 'title', 'link', and 'snippet'.
    """
    # News API endpoint
    endpoint = "https://newsapi.org/v2/everything"
    
    # Parameters for the search request
    params = {
        "q": keywords,
        "apiKey": "8719081539d34acd9c6cdb1c3417ae48",
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
                    "snippet": article.get('description', '')
                })
            
            return related_articles
        else:
            print("No news articles found.")
            return []
    else:
        print("Error:", response.status_code)
        return []


r=perform_newsapi_search("Linux great")
print(r)
