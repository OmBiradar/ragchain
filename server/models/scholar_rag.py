from scholarly import scholarly

def fetch_articles(query, max_results=5):
    """Fetch Google Scholar articles on a given topic"""
    # Search for publications matching the query
    search_results = scholarly.search_pubs(query)
    articles = []
    
    for _ in range(max_results):
        try:
            pub = next(search_results)
            bib = pub.get('bib', {})
            articles.append({
                'title': bib.get('title'),
                'authors': bib.get('author'),
                'abstract': bib.get('abstract'),
                'year': bib.get('pub_year')
            })
        except StopIteration:
            break
            
    return articles
