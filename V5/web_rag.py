from scholarly import scholarly

def fetch_google_scholar_articles(query, max_results=5):
    # Search for publications matching the query.
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

def main():
    query = 'machine learning'  # Replace with your desired query.
    articles = fetch_google_scholar_articles(query, max_results=10)
    
    for i, article in enumerate(articles, start=1):
        print(f"Article {i}:")
        print(f" Title: {article['title']}")
        print(f" Authors: {article['authors']}")
        print(f" Year: {article['year']}")
        print(f" Abstract: {article['abstract']}")
        print("-" * 80)

if __name__ == '__main__':
    main()