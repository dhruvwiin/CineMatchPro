"""
Shared test fixtures and configuration for CineMatchPro tests.

This module provides common fixtures used across all test files.
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import faiss
import os
from pathlib import Path


@pytest.fixture
def sample_movies_df():
    """
    Create a sample movie DataFrame for testing.
    Contains a diverse set of movies with various attributes.
    """
    return pd.DataFrame({
        'tconst': ['tt0133093', 'tt0234215', 'tt0242653', 'tt0111161', 'tt0068646', 'tt0468569'],
        'primaryTitle': ['The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions',
                         'The Shawshank Redemption', 'The Godfather', 'The Dark Knight'],
        'originalTitle': ['The Matrix', 'The Matrix Reloaded', 'The Matrix Revolutions',
                          'The Shawshank Redemption', 'The Godfather', 'The Dark Knight'],
        'startYear': [1999, 2003, 2003, 1994, 1972, 2008],
        'genres': ['Action,Sci-Fi', 'Action,Sci-Fi', 'Action,Sci-Fi',
                   'Drama', 'Crime,Drama', 'Action,Crime,Drama'],
        'averageRating': [8.7, 7.2, 6.8, 9.3, 9.2, 9.0],
        'numVotes': [1800000, 550000, 450000, 2500000, 1700000, 2400000],
        'runtimeMinutes': [136, 138, 129, 142, 175, 152],
        'popularity_score': [95.5, 82.3, 75.1, 98.2, 97.8, 96.5],
        'decade_str': ['1990s', '2000s', '2000s', '1990s', '1970s', '2000s'],
        'director_movie_count': [5, 5, 5, 8, 12, 10],
        'writer_movie_count': [3, 3, 3, 4, 6, 8],
        'cast1_movie_count': [25, 25, 25, 30, 45, 40],
        'cast2_movie_count': [20, 20, 20, 35, 40, 38],
        'cast3_movie_count': [18, 18, 18, 28, 42, 35]
    })


@pytest.fixture
def title_to_ids_mapping():
    """
    Create a title-to-IDs mapping for testing.
    Uses lowercase titles as keys.
    """
    return {
        'the matrix': ['tt0133093'],
        'the matrix reloaded': ['tt0234215'],
        'the matrix revolutions': ['tt0242653'],
        'the shawshank redemption': ['tt0111161'],
        'the godfather': ['tt0068646'],
        'the dark knight': ['tt0468569']
    }


@pytest.fixture
def mock_faiss_index(sample_movies_df):
    """
    Create a mock FAISS index for testing.
    Uses random normalized vectors.
    """
    n_movies = len(sample_movies_df)
    dimension = 50  # Smaller dimension for testing

    # Create random feature vectors and normalize them
    np.random.seed(42)  # For reproducibility
    vectors = np.random.randn(n_movies, dimension).astype('float32')
    # Normalize vectors (L2 norm)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return index


@pytest.fixture
def id_to_index_mapping(sample_movies_df):
    """
    Create ID-to-index mapping for testing.
    Maps tconst IDs to their FAISS index positions.
    """
    return pd.Series(
        range(len(sample_movies_df)),
        index=sample_movies_df['tconst'].values
    )


@pytest.fixture
def index_to_id_mapping(sample_movies_df):
    """
    Create index-to-ID mapping for testing.
    Maps FAISS index positions to tconst IDs.
    """
    return pd.Series(sample_movies_df['tconst'].values)


@pytest.fixture
def mock_model_data(sample_movies_df, title_to_ids_mapping, mock_faiss_index,
                    id_to_index_mapping, index_to_id_mapping):
    """
    Create a complete mock model_data dictionary for testing.
    This simulates the output of load_model_data().
    """
    return {
        'movie_metadata': sample_movies_df,
        'title_to_ids': title_to_ids_mapping,
        'faiss_index': mock_faiss_index,
        'id_to_index': id_to_index_mapping,
        'index_to_id': index_to_id_mapping,
        'scaler': None,  # Can be mocked if needed
        'tfidf_vectorizer': None  # Can be mocked if needed
    }


@pytest.fixture
def mock_omdb_response():
    """
    Create a mock successful OMDb API response.
    """
    return {
        'Response': 'True',
        'Title': 'The Matrix',
        'Year': '1999',
        'Rated': 'R',
        'Released': '31 Mar 1999',
        'Runtime': '136 min',
        'Genre': 'Action, Sci-Fi',
        'Director': 'Lana Wachowski, Lilly Wachowski',
        'Writer': 'Lilly Wachowski, Lana Wachowski',
        'Actors': 'Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss',
        'Plot': 'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
        'Language': 'English',
        'Country': 'United States, Australia',
        'Poster': 'https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg',
        'imdbRating': '8.7',
        'imdbVotes': '1,800,000',
        'imdbID': 'tt0133093'
    }


@pytest.fixture
def mock_omdb_not_found_response():
    """
    Create a mock OMDb API response for a movie not found.
    """
    return {
        'Response': 'False',
        'Error': 'Movie not found!'
    }


@pytest.fixture
def sample_csv_data(tmp_path):
    """
    Create a sample CSV file for testing model building.
    Returns the path to the created CSV file.

    Note: Creates 15 movies to satisfy TF-IDF min_df=5 requirement.
    """
    csv_path = tmp_path / "test_movies.csv"

    df = pd.DataFrame({
        'tconst': [f'tt{str(i).zfill(4)}' for i in range(1, 16)],
        'primaryTitle': [f'Test Movie {i}' for i in range(1, 16)],
        'originalTitle': [f'Test Movie {i}' for i in range(1, 16)],
        'startYear': [2000 + (i % 20) for i in range(1, 16)],
        'genres': [
            'Action,Comedy', 'Drama', 'Sci-Fi,Thriller', 'Romance,Comedy',
            'Action,Adventure', 'Horror,Thriller', 'Drama,Romance',
            'Comedy', 'Action,Sci-Fi', 'Drama,Thriller',
            'Animation,Comedy', 'Action,Drama', 'Sci-Fi,Adventure',
            'Romance,Drama', 'Comedy,Romance'
        ],
        'averageRating': [7.5 + (i % 3) * 0.5 for i in range(1, 16)],
        'numVotes': [10000 + i * 1000 for i in range(1, 16)],
        'runtimeMinutes': [110 + (i % 5) * 10 for i in range(1, 16)],
        'popularity_score': [70.0 + (i % 10) * 2 for i in range(1, 16)],
        'decade_str': ['2000s' if i < 10 else '2010s' for i in range(1, 16)],
        'director_movie_count': [5 + (i % 8) for i in range(1, 16)],
        'writer_movie_count': [4 + (i % 6) for i in range(1, 16)],
        'cast1_movie_count': [20 + (i % 10) for i in range(1, 16)],
        'cast2_movie_count': [18 + (i % 8) for i in range(1, 16)],
        'cast3_movie_count': [15 + (i % 7) for i in range(1, 16)]
    })

    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_model_dir(tmp_path, sample_movies_df, title_to_ids_mapping,
                   mock_faiss_index, id_to_index_mapping, index_to_id_mapping):
    """
    Create a complete mock model directory with all necessary files.
    Returns the path to the mock model directory.
    """
    model_dir = tmp_path / "model_faiss"
    model_dir.mkdir()

    # Save metadata CSV
    metadata_path = model_dir / "movie_metadata.csv"
    sample_movies_df.to_csv(metadata_path, index=False)

    # Save FAISS index
    index_path = model_dir / "faiss_index.idx"
    faiss.write_index(mock_faiss_index, str(index_path))

    # Save mappings
    with open(model_dir / "id_to_index.pkl", 'wb') as f:
        pickle.dump(id_to_index_mapping, f)

    with open(model_dir / "index_to_id.pkl", 'wb') as f:
        pickle.dump(index_to_id_mapping, f)

    with open(model_dir / "title_to_ids.pkl", 'wb') as f:
        pickle.dump(title_to_ids_mapping, f)

    return model_dir


@pytest.fixture
def mock_gemini_summary_response():
    """
    Create a mock Gemini AI summary response.
    """
    return "A thrilling sci-fi adventure where a computer hacker discovers the shocking truth about reality. Mind-bending action and philosophical questions collide in this groundbreaking film!"


@pytest.fixture
def mock_gemini_recommendations_response():
    """
    Create a mock Gemini AI recommendations response.
    """
    return """1. The Matrix
2. Blade Runner
3. Inception
4. Total Recall
5. Dark City"""
