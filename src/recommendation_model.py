import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import os

class MovieRecommendationNet(nn.Module):
    """Neural network for movie recommendations"""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super(MovieRecommendationNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class RecommendationModel:
    """Class to manage the recommendation model"""

    def __init__(self, config, session_id=None):
        self.config = config
        self.session_id = session_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data_loader = None
        self.user_profile = self._load_user_profile()
        
        print(f"Using device: {self.device}")

    def set_data_loader(self, data_loader):
        """Sets the data loader"""
        self.data_loader = data_loader

        # Initialize model with correct dimensions
        if self.data_loader:
            input_dim = self.data_loader.get_feature_dim()
            self.model = MovieRecommendationNet(input_dim).to(self.device)
            
            # Load model if it exists
            self._load_model()

    def _get_user_profile_path(self):
        """Gets the user profile file path for the session"""
        if self.session_id:
            user_paths = self.config.get_user_paths(self.session_id)
            return user_paths.get('user_profile_file', self.config.user_profile_file)
        return self.config.user_profile_file

    def _get_model_file_path(self):
        """Gets the model file path for the session"""
        if self.session_id:
            user_paths = self.config.get_user_paths(self.session_id)
            return user_paths.get('model_file')
        # Fallback for when there's no session
        return os.path.join(self.config.output_path, 'recommendation_model.pt')

    def _load_user_profile(self):
        """Loads the user profile"""
        profile_path = self._get_user_profile_path()
        try:
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    return json.load(f)
        except:
            pass

        return {
            'favorite_genres': {},
            'favorite_tags': {},
            'disliked_genres': {},
            'disliked_tags': {},
            'model_trained': False,
            'total_ratings': 0
        }

    def _save_user_profile(self):
        """Saves the user profile"""
        profile_path = self._get_user_profile_path()
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(profile_path), exist_ok=True)
            
            with open(profile_path, 'w') as f:
                json.dump(self.user_profile, f, indent=2)
        except Exception as e:
            print(f"Error saving user profile: {e}")

    def update_user_preferences(self, movie_id, rating):
        """Updates user preferences with a new rating"""
        if not self.data_loader:
            return

        movie = self.data_loader.get_movie_by_id(movie_id)
        if not movie:
            return

        self.user_profile['total_ratings'] += 1

        # Process genres
        if movie.get('genres'):
            genres = movie['genres'].split('|')
            for genre in genres:
                if rating == 1:  # Like
                    self.user_profile['favorite_genres'][genre] = \
                        self.user_profile['favorite_genres'].get(genre, 0) + 1
                else:  # Dislike
                    self.user_profile['disliked_genres'][genre] = \
                        self.user_profile['disliked_genres'].get(genre, 0) + 1

        # Process tags
        if movie.get('tags'):
            tags = [tag.strip() for tag in movie['tags'].split(',')]
            for tag in tags:
                if rating == 1:  # Like
                    self.user_profile['favorite_tags'][tag] = \
                        self.user_profile['favorite_tags'].get(tag, 0) + 1
                else:  # Dislike
                    self.user_profile['disliked_tags'][tag] = \
                        self.user_profile['disliked_tags'].get(tag, 0) + 1

        self._save_user_profile()

        # Retrain model based on configured frequency
        if self.user_profile['total_ratings'] % self.config.auto_train_frequency == 0:
            self.train_model()

    def train_model(self, user_state=None):
        """Trains the model with user data"""
        if not self.data_loader:
            print("Data loader not available")
            return

        # If user_state is not passed, create one for the session
        if user_state is None:
            from user_state import UserState
            user_state = UserState(self.config, self.session_id)

        # Get training data from user_state
        training_data = user_state.get_all_rated_movies()

        if len(training_data) < 3:
            print("Not enough data for training (minimum 3 ratings)")
            return

        print(f"Training model with {len(training_data)} ratings...")

        # Prepare data
        X, y = self._prepare_training_data(training_data)

        if X is None or len(X) == 0:
            print("Error preparing training data")
            return

        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)

        # Set up model if it doesn't exist
        if not self.model:
            input_dim = X_train.shape[1]
            self.model = MovieRecommendationNet(input_dim).to(self.device)

        # Training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.model.train()

        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.config.epochs}], Loss: {loss.item():.4f}')

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            accuracy = (test_predictions == y_test_tensor).float().mean()
            print(f'Model accuracy: {accuracy.item():.4f}')

        # Save model
        self._save_model()

        self.user_profile['model_trained'] = True
        self._save_user_profile()

        print("Training completed!")

    def _prepare_training_data(self, training_data):
        """Prepares data for training"""
        X = []
        y = []

        for item in training_data:
            features = self.data_loader.get_movie_features(item['movieId'])
            if features is not None and len(features) > 0:
                X.append(features)
                y.append(item['rating'])

        if len(X) == 0:
            return None, None

        return np.array(X), np.array(y)

    def get_recommendation(self, user_state):
        """Gets a personalized recommendation"""
        if not self.data_loader:
            return None

        # If model is not trained, use rule-based recommendations
        if not self.user_profile.get('model_trained', False):
            recommended_movie = self._get_rule_based_recommendation(user_state)
        else:
            # Use model for recommendations
            recommended_movie = self._get_model_based_recommendation(user_state)

        # Track the shown movie
        if recommended_movie and recommended_movie.get('title'):
            user_state.add_recently_shown(recommended_movie['title'])
            
        return recommended_movie

    def _get_rule_based_recommendation(self, user_state):
        """Rule-based recommendations"""
        # Get all movies
        all_movies = self.data_loader.get_all_movies()

        # Filter already rated movies and recently shown ones
        rated_titles = set()
        for movie in user_state.get_liked_movies() + user_state.get_disliked_movies():
            rated_titles.add(movie.get('title', ''))

        recently_shown_titles = set(user_state.get_recently_shown())

        unrated_movies = [m for m in all_movies
                         if m['title'] not in rated_titles and m['title'] not in recently_shown_titles]

        # If no unrated and not recently shown movies, show rated but not recent ones
        if not unrated_movies:
            unrated_movies = [m for m in all_movies if m['title'] not in recently_shown_titles]
            
        if not unrated_movies:
            return None
        
        # If there are preferences, recommend based on them
        if self.user_profile['favorite_genres']:
            favorite_genre = max(self.user_profile['favorite_genres'],
                               key=self.user_profile['favorite_genres'].get)

            # Search for movies with favorite genre
            matching_movies = [m for m in unrated_movies
                             if m.get('genres') and favorite_genre in m['genres']]

            if matching_movies:
                return np.random.choice(matching_movies)
        
        # Otherwise return a random movie
        return np.random.choice(unrated_movies)

    def _get_model_based_recommendation(self, user_state):
        """Model-based recommendations"""
        if not self.model:
            return self._get_rule_based_recommendation(user_state)
        
        # Get all unrated and not recently shown movies
        all_movies = self.data_loader.get_all_movies()
        rated_titles = set()
        for movie in user_state.get_liked_movies() + user_state.get_disliked_movies():
            rated_titles.add(movie.get('title', ''))
        
        recently_shown_titles = set(user_state.get_recently_shown())

        unrated_movies = [m for m in all_movies
                         if m['title'] not in rated_titles and m['title'] not in recently_shown_titles]

        # If there are no unrated and not recently shown movies, use only those not recently shown
        if not unrated_movies:
            unrated_movies = [m for m in all_movies if m['title'] not in recently_shown_titles]

        if not unrated_movies:
            return None

        # Calculate probabilities for all unrated movies
        movie_scores = []

        self.model.eval()
        with torch.no_grad():
            for movie in unrated_movies:
                features = self.data_loader.get_movie_features(movie['movieId'])
                if features is not None and len(features) > 0:
                    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                    score = self.model(features_tensor).item()
                    movie_scores.append((movie, score))

        if not movie_scores:
            return np.random.choice(unrated_movies)

        # Sort by score and take the top 10
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        top_movies = [item[0] for item in movie_scores[:10]]

        # Return a random movie among the top 10
        return np.random.choice(top_movies)
    
    def _save_model(self):
        """Saves the model"""
        if self.model:
            try:
                model_path = self._get_model_file_path()
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved in {model_path}")
            except Exception as e:
                print(f"Error in saving model: {e}")
    
    def _load_model(self):
        """Loads the model if it exists"""
        model_path = self._get_model_file_path()
        if os.path.exists(model_path) and self.model:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {e}")
