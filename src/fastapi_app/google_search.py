import os

import requests
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()


def google_search_function(search_query, exact_term=None):
    if not search_query:
        return []
    # Replace with your actual API key
    api_key = os.environ["GOOGLE_SEARCH_API_KEY"]

    # Custom search engine ID
    cx = os.environ["GOOGLE_SEARCH_ENGINE_ID"]

    # Construct the URL
    if exact_term:
        url = (
            f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={search_query}&exactTerms={exact_term}"
        )
    else:
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={search_query}"
    # Send the GET request
    response = requests.get(url)

    links = []
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        for item in data.get("items", []):
            link = item.get("link")
            if link:
                links.append(link)

        if len(links) < 1 and exact_term:
            # Exact term could not find results -> try again without
            return google_search_function(search_query=search_query, exact_term=None)

        return links
    else:
        return {"error": "Google search failed"}


if __name__ == "__main__":
    search_query = "ลดขนาดหน้าอก"
    result = google_search_function(search_query)
    print(result)
