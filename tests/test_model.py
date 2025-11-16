"""
Tests for model.py

This module tests the model building functionality including:
- Feature extraction
- FAISS index creation
- Model artifact saving
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
import os
import pickle
import faiss
from model import build_recommendation_model


class TestBuildRecommendationModel:
    """Tests for the build_recommendation_model function."""

    def test_build_model_success(self, sample_csv_data, tmp_path):
        """Test successful model building with valid data."""
        output_dir = tmp_path / "test_model_output"

        result = build_recommendation_model(str(sample_csv_data), str(output_dir))

        assert result is not None
        assert result == str(output_dir)
        assert os.path.exists(output_dir)

        # Check that all required files were created
        assert os.path.exists(output_dir / "faiss_index.idx")
        assert os.path.exists(output_dir / "movie_metadata.csv")
        assert os.path.exists(output_dir / "scaler.pkl")
        assert os.path.exists(output_dir / "tfidf_vectorizer.pkl")
        assert os.path.exists(output_dir / "id_to_index.pkl")
        assert os.path.exists(output_dir / "index_to_id.pkl")
        assert os.path.exists(output_dir / "title_to_ids.pkl")

    def test_model_artifacts_validity(self, sample_csv_data, tmp_path):
        """Test that created model artifacts are valid and loadable."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        # Load and verify FAISS index
        index = faiss.read_index(str(output_dir / "faiss_index.idx"))
        assert index.ntotal == 15  # We have 15 movies in sample data

        # Load and verify metadata
        metadata = pd.read_csv(output_dir / "movie_metadata.csv")
        assert len(metadata) == 15
        assert 'tconst' in metadata.columns
        assert 'primaryTitle' in metadata.columns

        # Load and verify mappings
        with open(output_dir / "id_to_index.pkl", 'rb') as f:
            id_to_index = pickle.load(f)
        assert len(id_to_index) == 15

        with open(output_dir / "index_to_id.pkl", 'rb') as f:
            index_to_id = pickle.load(f)
        assert len(index_to_id) == 15

        with open(output_dir / "title_to_ids.pkl", 'rb') as f:
            title_to_ids = pickle.load(f)
        assert len(title_to_ids) >= 15

    def test_id_mapping_consistency(self, sample_csv_data, tmp_path):
        """Test that ID mappings are consistent with each other."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        with open(output_dir / "id_to_index.pkl", 'rb') as f:
            id_to_index = pickle.load(f)

        with open(output_dir / "index_to_id.pkl", 'rb') as f:
            index_to_id = pickle.load(f)

        # Verify that mappings are inverses of each other
        for movie_id, index in id_to_index.items():
            assert index_to_id[index] == movie_id

    def test_build_model_missing_input_file(self, tmp_path):
        """Test that model building fails gracefully with missing input file."""
        output_dir = tmp_path / "test_model_output"
        nonexistent_file = tmp_path / "nonexistent.csv"

        result = build_recommendation_model(str(nonexistent_file), str(output_dir))

        assert result is None

    def test_build_model_creates_output_directory(self, sample_csv_data, tmp_path):
        """Test that model building creates output directory if it doesn't exist."""
        output_dir = tmp_path / "new_directory" / "nested" / "model"

        assert not os.path.exists(output_dir)

        result = build_recommendation_model(str(sample_csv_data), str(output_dir))

        assert result is not None
        assert os.path.exists(output_dir)

    def test_faiss_index_dimension(self, sample_csv_data, tmp_path):
        """Test that FAISS index has the correct dimension."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        index = faiss.read_index(str(output_dir / "faiss_index.idx"))

        # Dimension should be: TF-IDF features + numerical features
        # The exact dimension depends on the vocabulary and numerical columns
        assert index.d > 0  # Should have some dimension

    def test_feature_names_saved(self, sample_csv_data, tmp_path):
        """Test that feature names are saved correctly."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        features_path = output_dir / "feature_names.pkl"
        assert os.path.exists(features_path)

        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    def test_tfidf_vectorizer_properties(self, sample_csv_data, tmp_path):
        """Test that TF-IDF vectorizer is configured correctly."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        with open(output_dir / "tfidf_vectorizer.pkl", 'rb') as f:
            tfidf = pickle.load(f)

        assert tfidf.stop_words == 'english'
        assert tfidf.min_df == 5
        assert tfidf.max_df == 0.7

    def test_scaler_properties(self, sample_csv_data, tmp_path):
        """Test that StandardScaler is created and saved."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        with open(output_dir / "scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        # Scaler should have been fitted
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')

    def test_metadata_columns_preserved(self, sample_csv_data, tmp_path):
        """Test that important metadata columns are preserved in output."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        metadata = pd.read_csv(output_dir / "movie_metadata.csv")

        required_columns = ['tconst', 'primaryTitle', 'startYear', 'genres', 'averageRating']
        for col in required_columns:
            assert col in metadata.columns

    def test_build_with_missing_numerical_values(self, tmp_path):
        """Test model building handles missing numerical values."""
        # Create CSV with some missing values (need 20+ movies for TF-IDF min_df=5)
        csv_path = tmp_path / "movies_with_nulls.csv"

        n_movies = 20
        genres_pattern = ['Action,Drama', 'Drama,Comedy', 'Comedy,Romance', 'Thriller,Action', 'Romance,Drama']

        df = pd.DataFrame({
            'tconst': [f'tt{str(i).zfill(4)}' for i in range(1, n_movies + 1)],
            'primaryTitle': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'originalTitle': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'startYear': [2000 + (i % 20) if i % 3 != 0 else np.nan for i in range(n_movies)],  # Some missing
            'genres': [genres_pattern[i % len(genres_pattern)] for i in range(n_movies)],  # Repeated genres
            'averageRating': [7.0 + (i % 3) * 0.5 if i % 4 != 0 else np.nan for i in range(n_movies)],  # Some missing
            'numVotes': [10000 + i * 1000 if i % 5 != 0 else np.nan for i in range(n_movies)],  # Some missing
            'runtimeMinutes': [110 + (i % 6) * 5 for i in range(n_movies)],
            'popularity_score': [70.0 + (i % 10) * 2 for i in range(n_movies)],
            'decade_str': ['2000s' if i < 10 else '2010s' for i in range(n_movies)],
            'director_movie_count': [5 + (i % 8) for i in range(n_movies)],
            'writer_movie_count': [4 + (i % 6) for i in range(n_movies)],
            'cast1_movie_count': [20 + (i % 10) for i in range(n_movies)],
            'cast2_movie_count': [18 + (i % 8) for i in range(n_movies)],
            'cast3_movie_count': [15 + (i % 7) for i in range(n_movies)]
        })

        df.to_csv(csv_path, index=False)

        output_dir = tmp_path / "test_model_output"

        result = build_recommendation_model(str(csv_path), str(output_dir))

        # Should still succeed - missing values are filled with median
        assert result is not None
        assert os.path.exists(output_dir / "faiss_index.idx")

    def test_build_with_empty_genres(self, tmp_path):
        """Test model building handles empty/missing genres."""
        csv_path = tmp_path / "movies_empty_genres.csv"

        n_movies = 20
        # Mix of genres with some empty and NaN
        genres_list = []
        genres_pattern = ['Action,Drama', 'Drama,Comedy', 'Comedy,Romance', 'Thriller,Action', 'Romance,Drama']
        for i in range(n_movies):
            if i % 6 == 0:
                genres_list.append('')  # Empty string
            elif i % 7 == 0:
                genres_list.append(np.nan)  # NaN
            else:
                genres_list.append(genres_pattern[i % len(genres_pattern)])

        df = pd.DataFrame({
            'tconst': [f'tt{str(i).zfill(4)}' for i in range(1, n_movies + 1)],
            'primaryTitle': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'originalTitle': [f'Movie {i}' for i in range(1, n_movies + 1)],
            'startYear': [2000 + i for i in range(n_movies)],
            'genres': genres_list,
            'averageRating': [7.0 + (i % 3) * 0.5 for i in range(n_movies)],
            'numVotes': [10000 + i * 1000 for i in range(n_movies)],
            'runtimeMinutes': [110 + (i % 6) * 5 for i in range(n_movies)],
            'popularity_score': [70.0 + (i % 10) * 2 for i in range(n_movies)],
            'decade_str': ['2000s' if i < 10 else '2010s' for i in range(n_movies)],
            'director_movie_count': [5 + (i % 8) for i in range(n_movies)],
            'writer_movie_count': [4 + (i % 6) for i in range(n_movies)],
            'cast1_movie_count': [20 + (i % 10) for i in range(n_movies)],
            'cast2_movie_count': [18 + (i % 8) for i in range(n_movies)],
            'cast3_movie_count': [15 + (i % 7) for i in range(n_movies)]
        })

        df.to_csv(csv_path, index=False)

        output_dir = tmp_path / "test_model_output"

        result = build_recommendation_model(str(csv_path), str(output_dir))

        # Should handle empty genres gracefully
        assert result is not None

    def test_title_to_ids_lowercase(self, sample_csv_data, tmp_path):
        """Test that title_to_ids mapping uses lowercase titles."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        with open(output_dir / "title_to_ids.pkl", 'rb') as f:
            title_to_ids = pickle.load(f)

        # All keys should be lowercase
        for title in title_to_ids.keys():
            assert title == title.lower()

    def test_duplicate_titles_handling(self, tmp_path):
        """Test that duplicate titles are handled correctly in mapping."""
        csv_path = tmp_path / "movies_duplicates.csv"

        df = pd.DataFrame({
            'tconst': [f'tt{str(i).zfill(4)}' for i in range(1, 13)],
            'primaryTitle': ['The Movie', 'The Movie', 'Different Movie'] + [f'Movie {i}' for i in range(4, 13)],
            'originalTitle': ['The Movie', 'The Movie', 'Different Movie'] + [f'Movie {i}' for i in range(4, 13)],
            'startYear': [2000 + i for i in range(12)],
            'genres': ['Action,Drama', 'Drama,Comedy', 'Comedy,Romance', 'Thriller,Action', 'Romance,Drama', 'Horror,Thriller', 'Sci-Fi,Action', 'Adventure,Action', 'Animation,Comedy', 'Mystery,Thriller', 'Fantasy,Drama', 'Western,Action'],  # Repeated genres
            'averageRating': [7.5, 8.0, 6.5, 7.0, 8.5, 6.8, 7.2, 7.8, 8.2, 7.3, 7.7, 8.1],
            'numVotes': [10000 + i * 1000 for i in range(12)],
            'runtimeMinutes': [110 + i * 5 for i in range(12)],
            'popularity_score': [70.0 + i * 2 for i in range(12)],
            'decade_str': ['2000s' if i < 6 else '2010s' for i in range(12)],
            'director_movie_count': [5 + i for i in range(12)],
            'writer_movie_count': [4 + i for i in range(12)],
            'cast1_movie_count': [20 + i for i in range(12)],
            'cast2_movie_count': [18 + i for i in range(12)],
            'cast3_movie_count': [15 + i for i in range(12)]
        })

        df.to_csv(csv_path, index=False)

        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(csv_path), str(output_dir))

        with open(output_dir / "title_to_ids.pkl", 'rb') as f:
            title_to_ids = pickle.load(f)

        # 'the movie' should map to a list with both IDs
        assert 'the movie' in title_to_ids
        assert len(title_to_ids['the movie']) == 2
        assert 'tt0001' in title_to_ids['the movie']
        assert 'tt0002' in title_to_ids['the movie']


class TestModelDataIntegrity:
    """Tests for data integrity after model building."""

    def test_faiss_index_size_matches_metadata(self, sample_csv_data, tmp_path):
        """Test that FAISS index size matches number of movies in metadata."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        index = faiss.read_index(str(output_dir / "faiss_index.idx"))
        metadata = pd.read_csv(output_dir / "movie_metadata.csv")

        assert index.ntotal == len(metadata)

    def test_all_movie_ids_in_mappings(self, sample_csv_data, tmp_path):
        """Test that all movie IDs from metadata are in the mappings."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        metadata = pd.read_csv(output_dir / "movie_metadata.csv")

        with open(output_dir / "id_to_index.pkl", 'rb') as f:
            id_to_index = pickle.load(f)

        # All tconst values should be in id_to_index
        for tconst in metadata['tconst']:
            assert tconst in id_to_index.index

    def test_metadata_order_matches_faiss_index(self, sample_csv_data, tmp_path):
        """Test that metadata row order matches FAISS index positions."""
        output_dir = tmp_path / "test_model_output"

        build_recommendation_model(str(sample_csv_data), str(output_dir))

        metadata = pd.read_csv(output_dir / "movie_metadata.csv")

        with open(output_dir / "index_to_id.pkl", 'rb') as f:
            index_to_id = pickle.load(f)

        # For each position, the tconst in metadata should match index_to_id
        for i in range(len(metadata)):
            assert metadata.iloc[i]['tconst'] == index_to_id[i]
