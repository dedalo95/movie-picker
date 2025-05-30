import os
from flask import Flask, render_template, request, jsonify, session
from data_loader import DataLoader
from utils import TMDBApiEnhanced
from recommendation_model import RecommendationModel
from user_state import UserState
from config import Config
import json
import uuid

app = Flask(__name__, template_folder='../templates', static_folder='../static')

# Initialize components
config = Config()
app.secret_key = config.secret_key

data_loader = DataLoader(config)
tmdb_api = TMDBApiEnhanced(config, data_loader)

# Dictionary to maintain instances per session
user_states = {}
recommendation_models = {}

def get_session_id():
    """Gets or creates a unique session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_user_state():
    """Gets or creates UserState for the current session"""
    session_id = get_session_id()
    if session_id not in user_states:
        user_states[session_id] = UserState(config, session_id)
    return user_states[session_id]

def get_recommendation_model():
    """Gets or creates RecommendationModel for the current session"""
    session_id = get_session_id()
    if session_id not in recommendation_models:
        rec_model = RecommendationModel(config, session_id)
        rec_model.set_data_loader(data_loader)
        recommendation_models[session_id] = rec_model
    return recommendation_models[session_id]

@app.route('/')
def index():
    """Homepage - shows a movie to rate"""
    user_state = get_user_state()
    recommendation_model = get_recommendation_model()
    
    # Get recommendation based on user preferences
    recommended_movie = recommendation_model.get_recommendation(user_state)
    
    if recommended_movie is None:
        # If there are no recommendations, get a random movie excluding recently shown ones
        recently_shown = user_state.get_recently_shown()
        recommended_movie = data_loader.get_random_movie(exclude_titles=recently_shown)
        
        # Track the shown movie even when random
        if recommended_movie and recommended_movie.get('title'):
            user_state.add_recently_shown(recommended_movie['title'])
    
    # Get details from TMDB
    movie_details = tmdb_api.get_movie_details(recommended_movie['movieId'])
    
    return render_template('index.html', movie=movie_details)

@app.route('/like', methods=['POST'])
def like_movie():
    """Handles liking a movie"""
    data = request.get_json()
    user_state = get_user_state()
    recommendation_model = get_recommendation_model()
    
    # Add to liked
    user_state.add_liked_movie({
        'title': data['title'],
        'poster_path': data.get('poster_path'),
        'overview': data.get('overview'),
        'movieId': data.get('movieId'),
        'genres': data.get('genres'),
        'tags': data.get('tags')
    })
    
    # Remove from watchlist if present
    user_state.remove_from_watchlist(data['title'])
    
    # Update model with new preference
    recommendation_model.update_user_preferences(data['movieId'], 1)  # 1 = like
    
    return jsonify({'status': 'success'})

@app.route('/dislike', methods=['POST'])
def dislike_movie():
    """Handles disliking a movie"""
    data = request.get_json()
    user_state = get_user_state()
    recommendation_model = get_recommendation_model()
    
    # Add to disliked
    user_state.add_disliked_movie({
        'title': data['title'],
        'poster_path': data.get('poster_path'),
        'overview': data.get('overview'),
        'movieId': data.get('movieId'),
        'genres': data.get('genres'),
        'tags': data.get('tags')
    })
    
    # Update model with new preference
    recommendation_model.update_user_preferences(data['movieId'], 0)  # 0 = dislike
    
    return jsonify({'status': 'success'})

@app.route('/not_seen', methods=['POST'])
def not_seen():
    """Handles adding a movie to the watchlist"""
    data = request.get_json()
    user_state = get_user_state()
    
    # Add to watchlist
    user_state.add_to_watchlist({
        'title': data['title'],
        'poster_path': data.get('poster_path'),
        'overview': data.get('overview'),
        'movieId': data.get('movieId'),
        'genres': data.get('genres'),
        'tags': data.get('tags')
    })
    
    return jsonify({'status': 'success'})

@app.route('/liked')
def liked_movies():
    """Shows the list of liked movies"""
    movies = get_user_state().get_liked_movies()
    return render_template('liked.html', movies=movies)

@app.route('/watchlist')
def watchlist():
    """Shows the watchlist"""
    movies = get_user_state().get_watchlist()
    return render_template('watchlist.html', movies=movies)

@app.route('/stats')
def stats():
    """Shows statistics and AI model info"""
    return render_template('stats.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Endpoint to retrain the model"""
    try:
        user_state = get_user_state()
        recommendation_model = get_recommendation_model()
        recommendation_model.train_model(user_state)
        return jsonify({'status': 'success', 'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns user and model statistics"""
    user_state = get_user_state()
    recommendation_model = get_recommendation_model()
    
    user_stats = user_state.get_stats()
    model_stats = {
        'model_trained': recommendation_model.user_profile.get('model_trained', False),
        'total_ratings': recommendation_model.user_profile.get('total_ratings', 0),
        'auto_train_frequency': config.auto_train_frequency,
        'favorite_genres': recommendation_model.user_profile.get('favorite_genres', {}),
        'favorite_tags': dict(list(recommendation_model.user_profile.get('favorite_tags', {}).items())[:5])  # Top 5
    }
    return jsonify({**user_stats, **model_stats})

@app.route('/api/set_train_frequency', methods=['POST'])
def set_train_frequency():
    """Sets the automatic training frequency"""
    data = request.get_json()
    frequency = data.get('frequency', 5)
    
    if frequency < 1:
        return jsonify({'status': 'error', 'message': 'Frequency must be at least 1'})
    
    config.auto_train_frequency = frequency
    return jsonify({'status': 'success', 'message': f'Training frequency set to {frequency} ratings'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
