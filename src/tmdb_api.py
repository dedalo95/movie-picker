import requests
import time

class TMDBApi:
    """Class to interact with the TMDB API"""
    
    def __init__(self, config):
        self.config = config
        self.api_key = config.tmdb_api_key
        self.base_url = config.tmdb_base_url
        self.poster_base_url = config.poster_base_url
        
    def get_movie_details(self, movie_id):
        """Gets movie details from TMDB using the local movie ID"""
        try:
            # First, try to fetch using the direct ID (if it's a tmdbId)
            url = f"{self.base_url}/movie/{movie_id}"
            params = {'api_key': self.api_key, 'language': 'it-IT'}
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._format_movie_data(data)
            else:
                # If it doesn't work, search by title
                return self._search_by_title_fallback(movie_id)
                
        except Exception as e:
            print(f"Error getting movie details {movie_id}: {e}")
            return self._get_default_movie_data()
    
    def _search_by_title_fallback(self, movie_id):
        """Fallback: search the movie by title if the ID doesn't work"""
        try:
            # Here you should have access to the DataLoader to get the title
            # For now, return default data
            return self._get_default_movie_data()
        except:
            return self._get_default_movie_data()
    
    def search_movie_by_title(self, title):
        """Search for a movie by title on TMDB"""
        try:
            url = f"{self.base_url}/search/movie"
            params = {
                'api_key': self.api_key,
                'query': title,
                'language': 'it-IT'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    return self._format_movie_data(data['results'][0])
            
            return self._get_default_movie_data()
            
        except Exception as e:
            print(f"Error searching for movie '{title}': {e}")
            return self._get_default_movie_data()
    
    def _format_movie_data(self, tmdb_data):
        """Format the data received from TMDB"""
        poster_path = tmdb_data.get('poster_path')
        if poster_path:
            poster_url = f"{self.poster_base_url}{poster_path}"
        else:
            poster_url = '/static/no-poster.jpg'
        
        return {
            'poster_path': poster_url,
            'overview': tmdb_data.get('overview', 'Overview not available'),
            'release_date': tmdb_data.get('release_date', ''),
            'vote_average': tmdb_data.get('vote_average', 0),
            'vote_count': tmdb_data.get('vote_count', 0),
            'popularity': tmdb_data.get('popularity', 0),
            'tmdb_id': tmdb_data.get('id')
        }
    
    def _get_default_movie_data(self):
        """Returns default data when TMDB is unavailable"""
        return {
            'poster_path': '/static/no-poster.jpg',
            'overview': 'Overview not available',
            'release_date': '',
            'vote_average': 0,
            'vote_count': 0,
            'popularity': 0,
            'tmdb_id': None
        }
    
    def get_movie_by_imdb_id(self, imdb_id):
        """Gets a movie using the IMDB ID"""
        try:
            url = f"{self.base_url}/find/{imdb_id}"
            params = {
                'api_key': self.api_key,
                'external_source': 'imdb_id',
                'language': 'it-IT'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data['movie_results']:
                    return self._format_movie_data(data['movie_results'][0])
            
            return self._get_default_movie_data()
            
        except Exception as e:
            print(f"Error getting movie with IMDB ID {imdb_id}: {e}")
            return self._get_default_movie_data()
            
    def batch_get_movie_details(self, movie_ids, delay=0.1):
        """Gets details of multiple movies with rate limiting"""
        results = {}
        
        for movie_id in movie_ids:
            results[movie_id] = self.get_movie_details(movie_id)
            time.sleep(delay)  # Rate limiting to respect API limits
            
        return results
