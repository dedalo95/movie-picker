import os
import secrets
from dotenv import load_dotenv

class Config:
    """Class to manage application configuration"""
    
    def __init__(self):
        load_dotenv()
        
        # File paths
        self.dataset_path = os.getenv('DATASET_PATH', '/Users/federico.antosiano/Desktop/movie-picker/dataset')
        self.movies_csv = os.path.join(self.dataset_path, os.getenv('MOVIES_CSV', 'movies.csv'))
        self.links_csv = os.path.join(self.dataset_path, os.getenv('LINKS_CSV', 'links.csv'))
        
        # Base output path
        self.output_path = '/Users/federico.antosiano/Desktop/movie-picker/output'
        self.sessions_path = os.path.join(self.output_path, 'sessions')
        
        # Global files (fallback when there's no session)
        self.user_profile_file = os.path.join(self.output_path, 'user_profile.json')
        
        # Flask session config
        self.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))
        
        # TMDB API
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')
        self.poster_base_url = os.getenv('POSTER_BASE_URL', 'https://image.tmdb.org/t/p/w500')
        self.tmdb_base_url = 'https://api.themoviedb.org/3'
        
        # Model parameters
        self.embedding_dim = 50  # Embedding dimensions
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        self.auto_train_frequency = 5  # Automatic training every N ratings
        
        # Create necessary folders
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.sessions_path, exist_ok=True)
    
    def get_user_paths(self, session_id):
        """Returns user-specific paths based on session"""
        user_dir = os.path.join(self.sessions_path, session_id)
        os.makedirs(user_dir, exist_ok=True)
        
        return {
            'user_dir': user_dir,
            'liked_movies_file': os.path.join(user_dir, 'liked_movies.json'),
            'disliked_movies_file': os.path.join(user_dir, 'disliked_movies.json'),
            'watchlist_file': os.path.join(user_dir, 'watchlist.json'),
            'model_file': os.path.join(user_dir, 'recommendation_model.pt'),
            'user_profile_file': os.path.join(user_dir, 'user_profile.json')
        }
