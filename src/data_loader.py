import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import re

class DataLoader:
    """Class to load and preprocess movie data"""
    
    def __init__(self, config):
        self.config = config
        self.movies_df = None
        self.links_df = None
        self.genres_encoder = None
        self.tags_encoder = None
        self.load_data()
        
    def load_data(self):
        """Load CSV datasets"""
        try:
            self.movies_df = pd.read_csv(self.config.movies_csv)
            self.links_df = pd.read_csv(self.config.links_csv)
            
            # Merge dataframes
            self.movies_df = pd.merge(self.movies_df, self.links_df, on='movieId', how='left')
            
            # Preprocess genres and tags
            self._preprocess_genres_and_tags()
            
            print(f"Loaded {len(self.movies_df)} movies")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _preprocess_genres_and_tags(self):
        """Preprocess genres and tags for the model"""
        # Process genres
        genres_list = []
        for genres in self.movies_df['genres']:
            if pd.notna(genres):
                genres_list.append(genres.split('|'))
            else:
                genres_list.append([])
        
        # Process tags
        tags_list = []
        for tags in self.movies_df['tags']:
            if pd.notna(tags):
                # Remove commas and split
                tag_items = [tag.strip() for tag in tags.split(',')]
                tags_list.append(tag_items)
            else:
                tags_list.append([])
        
        # Create encoders for genres and tags
        self.genres_encoder = MultiLabelBinarizer()
        self.tags_encoder = MultiLabelBinarizer()
        
        # Fit encoders
        self.genres_encoded = self.genres_encoder.fit_transform(genres_list)
        self.tags_encoded = self.tags_encoder.fit_transform(tags_list)
        
        # Add to dataframe
        self.movies_df['genres_list'] = genres_list
        self.movies_df['tags_list'] = tags_list
        
    def get_random_movie(self, exclude_titles=None):
        """Returns a random movie, optionally excluding some titles"""
        if self.movies_df is not None and len(self.movies_df) > 0:
            if exclude_titles:
                # Filter movies excluding those in the list
                available_movies = self.movies_df[~self.movies_df['title'].isin(exclude_titles)]
                if len(available_movies) == 0:
                    # If all movies are excluded, ignore the exclusion
                    available_movies = self.movies_df
            else:
                available_movies = self.movies_df
                
            random_idx = np.random.randint(0, len(available_movies))
            movie = available_movies.iloc[random_idx]
            return {
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'tags': movie['tags'],
                'imdbId': movie.get('imdbId'),
                'tmdbId': movie.get('tmdbId')
            }
        return None
    
    def get_movie_by_id(self, movie_id):
        """Returns a movie by ID"""
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie_row) > 0:
            movie = movie_row.iloc[0]
            return {
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'tags': movie['tags'],
                'imdbId': movie.get('imdbId'),
                'tmdbId': movie.get('tmdbId')
            }
        return None
    
    def get_movie_features(self, movie_id):
        """Returns the features of a movie (encoded genres and tags)"""
        try:
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
            genres_features = self.genres_encoded[movie_idx]
            tags_features = self.tags_encoded[movie_idx]
            # Concatenate features
            return np.concatenate([genres_features, tags_features])
        except:
            # If movie is not found, return a zero vector
            total_features = len(self.genres_encoder.classes_) + len(self.tags_encoder.classes_)
            return np.zeros(total_features)
    
    def get_all_movies(self):
        """Returns all movies"""
        movies = []
        for _, movie in self.movies_df.iterrows():
            movies.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'tags': movie['tags'],
                'imdbId': movie.get('imdbId'),
                'tmdbId': movie.get('tmdbId')
            })
        return movies
    
    def get_feature_dim(self):
        """Returns the dimension of the feature vector"""
        if self.genres_encoder and self.tags_encoder:
            return len(self.genres_encoder.classes_) + len(self.tags_encoder.classes_)
        return 0
    
    def search_movies_by_title(self, title_query):
        """Search movies by title"""
        matches = self.movies_df[self.movies_df['title'].str.contains(title_query, case=False, na=False)]
        movies = []
        for _, movie in matches.iterrows():
            movies.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'tags': movie['tags'],
                'imdbId': movie.get('imdbId'),
                'tmdbId': movie.get('tmdbId')
            })
        return movies
