import json
import os
from datetime import datetime

class UserState:
    """Class to manage user state (likes, dislikes, watchlist) per session"""
    
    def __init__(self, config, session_id=None):
        self.config = config
        self.session_id = session_id
        self.user_paths = None
        self.liked_movies = []
        self.disliked_movies = []
        self.watchlist = []
        self.recently_shown = []
        
        if session_id:
            self.set_session(session_id)

    def set_session(self, session_id):
        """Sets the user session and loads specific data"""
        self.session_id = session_id
        self.user_paths = self.config.get_user_paths(session_id)
        self.load_user_data()

    def load_user_data(self):
        """Loads user data from session-specific JSON files"""
        if not self.user_paths:
            return

        try:
            if os.path.exists(self.user_paths['liked_movies_file']):
                with open(self.user_paths['liked_movies_file'], 'r', encoding='utf-8') as f:
                    self.liked_movies = json.load(f)
        except Exception as e:
            print(f"Error loading liked movies for session {self.session_id}: {e}")
            self.liked_movies = []
            
        try:
            if os.path.exists(self.user_paths['disliked_movies_file']):
                with open(self.user_paths['disliked_movies_file'], 'r', encoding='utf-8') as f:
                    self.disliked_movies = json.load(f)
        except Exception as e:
            print(f"Error loading disliked movies for session {self.session_id}: {e}")
            self.disliked_movies = []
            
        try:
            if os.path.exists(self.user_paths['watchlist_file']):
                with open(self.user_paths['watchlist_file'], 'r', encoding='utf-8') as f:
                    self.watchlist = json.load(f)
        except Exception as e:
            print(f"Error loading watchlist for session {self.session_id}: {e}")
            self.watchlist = []

    def save_user_data(self):
        """Saves user data to session-specific JSON files"""
        if not self.user_paths:
            return
            
        # Create user directory if it doesn't exist
        os.makedirs(os.path.dirname(self.user_paths['liked_movies_file']), exist_ok=True)
            
        try:
            with open(self.user_paths['liked_movies_file'], 'w', encoding='utf-8') as f:
                json.dump(self.liked_movies, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving liked movies for session {self.session_id}: {e}")
            
        try:
            with open(self.user_paths['disliked_movies_file'], 'w', encoding='utf-8') as f:
                json.dump(self.disliked_movies, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving disliked movies for session {self.session_id}: {e}")

        try:
            with open(self.user_paths['watchlist_file'], 'w', encoding='utf-8') as f:
                json.dump(self.watchlist, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving watchlist for session {self.session_id}: {e}")

    def add_liked_movie(self, movie):
        """Adds a movie to liked movies"""
        movie['timestamp'] = datetime.now().isoformat()
        self.liked_movies.append(movie)
        self.save_user_data()

    def add_disliked_movie(self, movie):
        """Adds a movie to disliked movies"""
        movie['timestamp'] = datetime.now().isoformat()
        self.disliked_movies.append(movie)
        self.save_user_data()

    def add_to_watchlist(self, movie):
        """Adds a movie to the watchlist"""
        # Don't add if already present in likes or dislikes
        liked_titles = [m.get('title') for m in self.liked_movies]
        disliked_titles = [m.get('title') for m in self.disliked_movies]
        
        if movie.get('title') not in liked_titles and movie.get('title') not in disliked_titles:
            movie['timestamp'] = datetime.now().isoformat()
            self.watchlist.append(movie)
            self.save_user_data()

    def remove_from_watchlist(self, title):
        """Removes a movie from the watchlist"""
        self.watchlist = [m for m in self.watchlist if m.get('title') != title]
        self.save_user_data()

    def is_movie_rated(self, title):
        """Checks if a movie has already been rated (like or dislike)"""
        liked_titles = [m.get('title') for m in self.liked_movies]
        disliked_titles = [m.get('title') for m in self.disliked_movies]
        return title in liked_titles or title in disliked_titles

    def get_liked_movies(self):
        """Returns the list of liked movies"""
        return self.liked_movies

    def get_disliked_movies(self):
        """Returns the list of disliked movies"""
        return self.disliked_movies

    def get_watchlist(self):
        """Returns the watchlist"""
        return self.watchlist

    def get_all_rated_movies(self):
        """Returns all rated movies (liked + disliked) with rating"""
        rated_movies = []
        
        for movie in self.liked_movies:
            movie_data = movie.copy()
            movie_data['rating'] = 1  # 1 = like
            rated_movies.append(movie_data)
        
        for movie in self.disliked_movies:
            movie_data = movie.copy()
            movie_data['rating'] = 0  # 0 = dislike
            rated_movies.append(movie_data)
        
        return rated_movies
    
    def get_stats(self):
        """Returns user statistics"""
        return {
            'total_liked': len(self.liked_movies),
            'total_disliked': len(self.disliked_movies),
            'total_watchlist': len(self.watchlist),
            'total_rated': len(self.liked_movies) + len(self.disliked_movies),
            'session_id': self.session_id
        }
    
    def add_recently_shown(self, movie_title):
        """Adds a movie to the recently shown movies list"""
        if movie_title not in self.recently_shown:
            self.recently_shown.append(movie_title)
            # Keep only the last 20 movies to prevent the list from growing too large
            if len(self.recently_shown) > 20:
                self.recently_shown.pop(0)
    
    def get_recently_shown(self):
        """Returns the list of recently shown movies"""
        return self.recently_shown
    
    def is_recently_shown(self, movie_title):
        """Checks if a movie has been shown recently"""
        return movie_title in self.recently_shown
