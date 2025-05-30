import os
import json
import requests
from datetime import datetime, timedelta

class TMDBCache:
    """Simple cache for TMDB calls"""
    
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_duration = timedelta(hours=24)  # Cache valid for 24 hours
    
    def get_cache_path(self, key):
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, key):
        cache_path = self.get_cache_path(key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                # Check if the cache is still valid
                cached_time = datetime.fromisoformat(data['timestamp'])
                if datetime.now() - cached_time < self.cache_duration:
                    return data['content']
            except:
                pass
        return None
    
    def set(self, key, content):
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'content': content
                }, f)
        except:
            pass

class TMDBApiEnhanced:
    """Enhanced version of the TMDB class with cache and intelligent fallback"""
    
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.api_key = config.tmdb_api_key
        self.base_url = config.tmdb_base_url
        self.poster_base_url = config.poster_base_url
        self.cache = TMDBCache(os.path.join(config.output_path, 'tmdb_cache'))
        
    def get_movie_details(self, movie_id):
        """Gets the details of a movie with cache and intelligent fallback"""
        
        # First check the cache
        cache_key = f"movie_{movie_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Get the movie from the data loader
        movie = self.data_loader.get_movie_by_id(movie_id)
        if not movie:
            return self._get_default_movie_data()
        
        # Try different strategies to get data from TMDB
        tmdb_data = None
        
        # 1. Try with TMDB ID if available
        if movie.get('tmdbId') and str(movie['tmdbId']).isdigit():
            tmdb_data = self._get_by_tmdb_id(movie['tmdbId'])
        
        # 2. Try with IMDB ID if available
        if not tmdb_data and movie.get('imdbId'):
            imdb_id_formatted = f"tt{str(movie['imdbId']).zfill(7)}"
            tmdb_data = self._get_by_imdb_id(imdb_id_formatted)
        
        # 3. Try searching by title
        if not tmdb_data:
            tmdb_data = self._search_by_title(movie['title'])
        
        # Combine local data with TMDB data
        result = self._combine_movie_data(movie, tmdb_data)
        
        # Save to cache
        self.cache.set(cache_key, result)
        
        return result
    
    def _get_by_tmdb_id(self, tmdb_id):
        """Fetch a movie using TMDB ID"""
        try:
            url = f"{self.base_url}/movie/{tmdb_id}"
            params = {'api_key': self.api_key, 'language': 'en-US'}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching TMDB ID {tmdb_id}: {e}")
        return None
    
    def _get_by_imdb_id(self, imdb_id):
        """Gets a movie using the IMDB ID"""
        try:
            url = f"{self.base_url}/find/{imdb_id}"
            params = {
                'api_key': self.api_key,
                'external_source': 'imdb_id',
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('movie_results'):
                    return data['movie_results'][0]
        except Exception as e:
            print(f"Error fetching IMDB ID {imdb_id}: {e}")
        return None
    
    def _search_by_title(self, title):
        """Search for a movie by title"""
        try:
            # Clean the title by removing the year if present
            clean_title = title.split('(')[0].strip()
            
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': clean_title,
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    # Take the first most relevant result
                    return data['results'][0]
        except Exception as e:
            print(f"Error searching by title '{title}': {e}")
        return None
    
    def _combine_movie_data(self, local_movie, tmdb_data):
        """Combines local data with TMDB data"""
        result = {
            'title': local_movie['title'],
            'genres': local_movie.get('genres', ''),
            'tags': local_movie.get('tags', ''),
            'movieId': local_movie['movieId'],
            'poster_path': '/static/no-poster.jpg',
            'overview': 'Plot not available',
            'release_date': '',
            'vote_average': 0,
            'vote_count': 0,
            'popularity': 0
        }
        
        if tmdb_data:
            # Add data from TMDB
            poster_path = tmdb_data.get('poster_path')
            if poster_path:
                result['poster_path'] = f"{self.poster_base_url}{poster_path}"
            
            result.update({
                'overview': tmdb_data.get('overview', 'Plot not available'),
                'release_date': tmdb_data.get('release_date', ''),
                'vote_average': tmdb_data.get('vote_average', 0),
                'vote_count': tmdb_data.get('vote_count', 0),
                'popularity': tmdb_data.get('popularity', 0),
                'tmdb_id': tmdb_data.get('id')
            })
        
        return result
    
    def _get_default_movie_data(self):
        """Returns default data"""
        return {
            'poster_path': '/static/no-poster.jpg',
            'overview': 'Plot not available',
            'release_date': '',
            'vote_average': 0,
            'vote_count': 0,
            'popularity': 0,
            'tmdb_id': None
        }

def extract_year_from_title(title):
    """Extracts the year from the movie title"""
    import re
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

def clean_title(title):
    """Cleans the title by removing the year and special characters"""
    import re
    # Remove the year
    title = re.sub(r'\(\d{4}\)', '', title).strip()
    # Remove extra special characters
    title = re.sub(r'[^\w\s]', ' ', title)
    # Remove multiple spaces
    title = re.sub(r'\s+', ' ', title).strip()
    return title
