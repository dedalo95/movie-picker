# Movie Picker

Movie Picker is a Tinder-style application for movies that integrates AI recommendations to help users find their next favorite movie.

## Features

- Swipe-style interface for movie selection.
- AI-powered movie recommendations.
- Integration with TMDB for movie posters and metadata.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dedalo95/movie-picker.git
   cd movie-picker
   ```
2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```
3. Set up the environment variables in the `.env` file. The default dataset path (`DATASET_PATH`) is set for Docker usage. If running locally, update it to match your local path.

   - Create a `.env` file in the root directory.
   - Add the following variables:
     ```properties
     TMDB_API_KEY=your_tmdb_api_key
     DATASET_PATH=/path/to/dataset
     MOVIES_CSV=movies.csv
     LINKS_CSV=links.csv
     POSTER_BASE_URL=https://image.tmdb.org/t/p/w500
     ```

## Usage

### Running Locally

1. Start the application:

   ```bash
   poetry run python main.py
   ```
2. Access the application at `http://localhost:5001`.

### Running with Docker

1. Build the Docker image:

   ```bash
   docker build -t movie-picker .
   ```
2. Run the Docker container:

   ```bash
   docker run -p 5001:5001 --env-file .env movie-picker
   ```
3. Access the application at `http://localhost:5001`.

Note: The default dataset path in the `.env` file (`/app/dataset`) is configured for Docker usage. Ensure the dataset files are correctly placed in the `dataset` directory before building the Docker image.

## AI Model

The Movie Picker application leverages cutting-edge AI models to deliver personalized movie recommendations. The model is built using deep learning frameworks such as PyTorch and Keras, ensuring high performance and scalability. Key aspects of the AI model include:

### Features

- **Collaborative Filtering**: Utilizes user-based and item-based collaborative filtering techniques to recommend movies based on user preferences and behavior.
- **Content-Based Filtering**: Analyzes movie metadata from the dataset, including genres and tags, to suggest movies similar to those the user likes. For example, if a user enjoys "Toy Story (1995)" with tags like "Epic" and "Action-Packed," the model can recommend other movies with similar tags and genres.
- **Hybrid Approach**: Combines collaborative and content-based filtering for more accurate and diverse recommendations.
- **Neural Network Architecture**: Employs advanced neural networks optimized for recommendation tasks, including embeddings and attention mechanisms.

### Training

The model is trained on a comprehensive dataset of movie preferences, ratings, and metadata. It uses:

- **Supervised Learning**: For predicting user preferences based on historical data.
- **Continuous Learning**: Adapts to user interactions in real-time, improving recommendations dynamically.

### Benefits

- **Personalization**: Tailors recommendations to individual users.
- **Scalability**: Handles large datasets and user bases efficiently.
- **Adaptability**: Learns from user feedback to refine predictions.

By integrating this AI model, Movie Picker ensures a unique and engaging experience for users, helping them discover movies they will love.

## Dataset

The Movie Picker application uses a comprehensive dataset of movies to provide accurate recommendations. The dataset includes:

- **Movie Metadata**: Information such as genres, tags, and titles.
- **User Preferences**: Historical data on user interactions and ratings.

### Example

Here is a sample from the dataset:

```csv
movieId,title,genres,tags
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy,"Toy, Story, Epic, Action-Packed, Exciting"
2,Jumanji (1995),Adventure|Children|Fantasy,"Jumanji, Epic, Action-Packed, Exciting, Family-Friendly"
```

This dataset is used to train the AI model and generate personalized recommendations.

## Requirements

- Python 3.10 or higher
- Flask
- NumPy
- Keras
- Python-dotenv
- Pandas
- Requests
- Torch
- Torchvision
- Scikit-learn

## Why Poetry?

Poetry is used in this project as the dependency management and packaging tool because it provides a streamlined and modern approach to handling Python projects. Key benefits include:

- **Dependency Resolution**: Poetry ensures that all dependencies are resolved efficiently and avoids conflicts.
- **Environment Isolation**: It creates virtual environments automatically, ensuring a clean and isolated development environment.
- **Ease of Use**: With simple commands, Poetry makes it easy to add, update, or remove dependencies.
- **Reproducibility**: The `poetry.lock` file guarantees that the same dependency versions are used across different environments.
- **Build and Publish**: Poetry simplifies the process of building and publishing Python packages.

By using Poetry, the Movie Picker project benefits from better dependency management and a more robust development workflow.

## User Interface

The Movie Picker application features an intuitive and engaging user interface with the following pages:

- **Home Page**: Displays the current movie with options to like, dislike, or add to the watchlist. Includes navigation buttons for accessing liked movies, watchlist, and statistics.
- **Liked Movies Page**: Shows a list of movies the user has liked, including their posters and overviews.
- **Watchlist Page**: Displays movies the user has marked as "not seen" with options to like or dislike them directly.
- **Statistics Page**: Provides insights into user preferences, favorite genres, top tags, and AI model status. Includes controls for training the model and adjusting training frequency.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
