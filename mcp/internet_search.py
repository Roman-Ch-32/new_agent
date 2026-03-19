# mcp/internet_search.py
"""Internet Search — Поиск в интернете"""

import requests
from typing import List, Dict, Any


class InternetSearch:
    """Инструменты для поиска в интернете"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Поиск через DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))

            return [
                {
                    'title': r.get('title', ''),
                    'url': r.get('href', ''),
                    'snippet': r.get('body', ''),
                    'source': 'DuckDuckGo'
                }
                for r in results
            ]
        except ImportError:
            return [{'error': 'Установите: pip install duckduckgo-search'}]
        except Exception as e:
            return [{'error': str(e)}]

    def search_github(self, query: str, language: str = 'C++') -> List[Dict[str, str]]:
        """Поиск кода на GitHub"""
        url = 'https://api.github.com/search/code'
        headers = {'Accept': 'application/vnd.github.v3+json'}
        params = {
            'q': f'{query} language:{language}',
            'per_page': 5
        }

        try:
            response = self.session.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return [
                {
                    'name': item.get('name', ''),
                    'path': item.get('path', ''),
                    'url': item.get('html_url', ''),
                    'repository': item.get('repository', {}).get('full_name', ''),
                    'source': 'GitHub'
                }
                for item in data.get('items', [])
            ]
        except Exception as e:
            return [{'error': str(e)}]