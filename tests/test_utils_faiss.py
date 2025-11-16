"""
Tests for utils_faiss.py

This module tests all functions in the utils_faiss module including:
- Data loading functions
- Recommendation engine functions
- External API integration functions
"""

import pytest
import pandas as pd
import numpy as np
import os
import pickle
import responses
import requests
from unittest.mock import Mock, patch, MagicMock
import utils_faiss as utils
import faiss


class TestLoadModelData:
    """Tests for the load_model_data function."""

    def test_load_model_data_success(self, mock_model_dir):
        """Test successful loading of model data."""
        model_data = utils.load_model_data(str(mock_model_dir))

        assert model_data is not None
        assert 'movie_metadata' in model_data
        assert 'faiss_index' in model_data
        assert 'id_to_index' in model_data
        assert 'index_to_id' in model_data
        assert 'title_to_ids' in model_data
        assert isinstance(model_data['movie_metadata'], pd.DataFrame)
        assert len(model_data['movie_metadata']) > 0

    def test_load_model_data_missing_metadata(self, tmp_path):
        """Test loading fails gracefully when metadata CSV is missing."""
        model_dir = tmp_path / "incomplete_model"
        model_dir.mkdir()

        model_data = utils.load_model_data(str(model_dir))
        assert model_data is None

    def test_load_model_data_missing_index(self, tmp_path, sample_movies_df):
        """Test loading fails when FAISS index is missing."""
        model_dir = tmp_path / "incomplete_model"
        model_dir.mkdir()

        # Save only metadata
        metadata_path = model_dir / "movie_metadata.csv"
        sample_movies_df.to_csv(metadata_path, index=False)

        model_data = utils.load_model_data(str(model_dir))
        assert model_data is None

    def test_load_model_data_missing_mappings(self, tmp_path, sample_movies_df, mock_faiss_index):
        """Test loading fails when mapping files are missing."""
        model_dir = tmp_path / "incomplete_model"
        model_dir.mkdir()

        # Save metadata and index but not mappings
        metadata_path = model_dir / "movie_metadata.csv"
        sample_movies_df.to_csv(metadata_path, index=False)

        index_path = model_dir / "faiss_index.idx"
        faiss.write_index(mock_faiss_index, str(index_path))

        model_data = utils.load_model_data(str(model_dir))
        assert model_data is None

    def test_load_model_data_nonexistent_directory(self):
        """Test loading from a directory that doesn't exist."""
        model_data = utils.load_model_data("/nonexistent/path/to/model")
        assert model_data is None


class TestGetMovieByTitle:
    """Tests for the get_movie_by_title function."""

    def test_exact_title_match(self, mock_model_data):
        """Test exact title matching (case-insensitive)."""
        result = utils.get_movie_by_title("The Matrix", mock_model_data)

        assert not result.empty
        assert len(result) >= 1
        assert result.iloc[0]['primaryTitle'] == 'The Matrix'
        assert result.iloc[0]['tconst'] == 'tt0133093'

    def test_case_insensitive_match(self, mock_model_data):
        """Test that title matching is case-insensitive."""
        result = utils.get_movie_by_title("the matrix", mock_model_data)

        assert not result.empty
        assert result.iloc[0]['primaryTitle'] == 'The Matrix'

    def test_partial_title_match(self, mock_model_data):
        """Test partial title matching."""
        result = utils.get_movie_by_title("Matrix", mock_model_data)

        assert not result.empty
        # Should find all Matrix movies
        assert len(result) >= 1

    def test_nonexistent_title(self, mock_model_data):
        """Test that non-existent title returns empty DataFrame."""
        result = utils.get_movie_by_title("Nonexistent Movie 12345", mock_model_data)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_empty_model_data(self):
        """Test behavior with None model_data."""
        result = utils.get_movie_by_title("The Matrix", None)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_missing_title_to_ids(self, sample_movies_df):
        """Test behavior when title_to_ids is missing from model_data."""
        incomplete_model_data = {
            'movie_metadata': sample_movies_df
            # Missing 'title_to_ids'
        }
        result = utils.get_movie_by_title("The Matrix", incomplete_model_data)

        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestGetMovieRecommendations:
    """Tests for the get_movie_recommendations function."""

    def test_valid_movie_id_returns_recommendations(self, mock_model_data):
        """Test that a valid movie ID returns recommendations."""
        movie_id = 'tt0133093'  # The Matrix
        recommendations = utils.get_movie_recommendations(movie_id, mock_model_data, n=5)

        assert not recommendations.empty
        assert len(recommendations) <= 5
        assert 'similarity' in recommendations.columns
        assert 'tconst' in recommendations.columns
        # The original movie should not be in recommendations
        assert movie_id not in recommendations['tconst'].values

    def test_invalid_movie_id(self, mock_model_data):
        """Test that an invalid movie ID returns empty DataFrame."""
        invalid_id = 'tt9999999'
        recommendations = utils.get_movie_recommendations(invalid_id, mock_model_data, n=5)

        assert isinstance(recommendations, pd.DataFrame)
        assert recommendations.empty

    def test_recommendations_are_sorted_by_similarity(self, mock_model_data):
        """Test that recommendations are sorted by similarity score."""
        movie_id = 'tt0133093'
        recommendations = utils.get_movie_recommendations(movie_id, mock_model_data, n=5)

        if not recommendations.empty:
            # Check that similarity scores are in descending order
            similarities = recommendations['similarity'].values
            assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))

    def test_empty_model_data(self):
        """Test behavior with None model_data."""
        recommendations = utils.get_movie_recommendations('tt0133093', None, n=5)

        assert isinstance(recommendations, pd.DataFrame)
        assert recommendations.empty

    def test_missing_required_keys(self, sample_movies_df):
        """Test behavior when required keys are missing from model_data."""
        incomplete_model_data = {
            'movie_metadata': sample_movies_df
            # Missing other required keys
        }
        recommendations = utils.get_movie_recommendations('tt0133093', incomplete_model_data, n=5)

        assert isinstance(recommendations, pd.DataFrame)
        assert recommendations.empty

    def test_number_of_recommendations(self, mock_model_data):
        """Test requesting different numbers of recommendations."""
        movie_id = 'tt0133093'

        # Request 3 recommendations
        recs_3 = utils.get_movie_recommendations(movie_id, mock_model_data, n=3)
        assert len(recs_3) <= 3

        # Request 1 recommendation
        recs_1 = utils.get_movie_recommendations(movie_id, mock_model_data, n=1)
        assert len(recs_1) <= 1


class TestGetRecommendationsByTitle:
    """Tests for the get_recommendations_by_title function."""

    def test_valid_title_returns_recommendations(self, mock_model_data):
        """Test that a valid title returns recommendations and matched movie."""
        recommendations, matched_movie = utils.get_recommendations_by_title(
            "The Matrix", mock_model_data, n=5
        )

        assert not recommendations.empty
        assert matched_movie is not None
        assert matched_movie['primaryTitle'] == 'The Matrix'
        assert len(recommendations) <= 5

    def test_nonexistent_title(self, mock_model_data):
        """Test that non-existent title returns empty recommendations."""
        recommendations, matched_movie = utils.get_recommendations_by_title(
            "Nonexistent Movie", mock_model_data, n=5
        )

        assert isinstance(recommendations, pd.DataFrame)
        assert recommendations.empty
        assert matched_movie is None

    def test_multiple_matches_picks_most_popular(self, mock_model_data):
        """Test that when multiple movies match, it picks the most popular one."""
        # Add duplicate title with different votes
        mock_model_data['movie_metadata'] = pd.concat([
            mock_model_data['movie_metadata'],
            pd.DataFrame({
                'tconst': ['tt9999999'],
                'primaryTitle': ['The Matrix'],
                'startYear': [1999],
                'genres': ['Action'],
                'averageRating': [8.0],
                'numVotes': [100],  # Much lower than original
                'runtimeMinutes': [120]
            })
        ])
        mock_model_data['title_to_ids']['the matrix'].append('tt9999999')

        recommendations, matched_movie = utils.get_recommendations_by_title(
            "The Matrix", mock_model_data, n=5
        )

        # Should pick the one with higher votes
        assert matched_movie['numVotes'] > 1000000


class TestGetTopMoviesByGenre:
    """Tests for the get_top_movies_by_genre function."""

    def test_filter_by_specific_genre(self, mock_model_data):
        """Test filtering movies by a specific genre."""
        top_movies = utils.get_top_movies_by_genre(
            'Action', (1990, 2010), mock_model_data, n=20
        )

        assert not top_movies.empty
        # Check that all movies contain 'Action' in genres
        for genres in top_movies['genres']:
            assert 'Action' in genres

    def test_filter_by_all_genres(self, mock_model_data):
        """Test filtering with 'All' genre returns all movies in year range."""
        top_movies = utils.get_top_movies_by_genre(
            'All', (1990, 2010), mock_model_data, n=20
        )

        assert not top_movies.empty
        # Should include movies from the specified year range
        assert all((1990 <= year <= 2010) for year in top_movies['startYear'])

    def test_filter_by_year_range(self, mock_model_data):
        """Test that year range filtering works correctly."""
        top_movies = utils.get_top_movies_by_genre(
            'All', (2000, 2005), mock_model_data, n=20
        )

        assert not top_movies.empty
        # All movies should be in the year range
        assert all((2000 <= year <= 2005) for year in top_movies['startYear'])

    def test_no_movies_in_range(self, mock_model_data):
        """Test when no movies match the genre and year criteria."""
        top_movies = utils.get_top_movies_by_genre(
            'Horror', (1800, 1850), mock_model_data, n=20
        )

        assert isinstance(top_movies, pd.DataFrame)
        # Should be empty or have very few results
        # (depending on test data)

    def test_limited_results(self, mock_model_data):
        """Test that results are limited to n movies."""
        top_movies = utils.get_top_movies_by_genre(
            'All', (1970, 2010), mock_model_data, n=3
        )

        assert len(top_movies) <= 3

    def test_empty_model_data(self):
        """Test behavior with None model_data."""
        top_movies = utils.get_top_movies_by_genre('Action', (1990, 2010), None, n=20)

        assert isinstance(top_movies, pd.DataFrame)
        assert top_movies.empty


class TestFetchOmdbDetails:
    """Tests for the fetch_omdb_details function."""

    @responses.activate
    def test_fetch_by_imdb_id_success(self, mock_omdb_response):
        """Test successful fetch using IMDb ID."""
        responses.add(
            responses.GET,
            'http://www.omdbapi.com/',
            json=mock_omdb_response,
            status=200
        )

        result = utils.fetch_omdb_details('test_api_key', imdb_id='tt0133093')

        assert result is not None
        assert result['Title'] == 'The Matrix'
        assert result['imdbID'] == 'tt0133093'
        assert result['Response'] == 'True'

    @responses.activate
    def test_fetch_by_title_and_year_success(self, mock_omdb_response):
        """Test successful fetch using title and year."""
        responses.add(
            responses.GET,
            'http://www.omdbapi.com/',
            json=mock_omdb_response,
            status=200
        )

        result = utils.fetch_omdb_details('test_api_key', title='The Matrix', year=1999)

        assert result is not None
        assert result['Title'] == 'The Matrix'

    @responses.activate
    def test_fetch_movie_not_found(self, mock_omdb_not_found_response):
        """Test handling when movie is not found."""
        responses.add(
            responses.GET,
            'http://www.omdbapi.com/',
            json=mock_omdb_not_found_response,
            status=200
        )

        result = utils.fetch_omdb_details('test_api_key', imdb_id='tt9999999')

        assert result is None

    @responses.activate
    def test_fetch_timeout(self):
        """Test handling of request timeout."""
        responses.add(
            responses.GET,
            'http://www.omdbapi.com/',
            body=requests.exceptions.Timeout()
        )

        result = utils.fetch_omdb_details('test_api_key', imdb_id='tt0133093')

        assert result is None

    @responses.activate
    def test_fetch_network_error(self):
        """Test handling of network errors."""
        responses.add(
            responses.GET,
            'http://www.omdbapi.com/',
            body=requests.exceptions.RequestException('Network error')
        )

        result = utils.fetch_omdb_details('test_api_key', imdb_id='tt0133093')

        assert result is None

    def test_fetch_no_api_key(self):
        """Test that function returns None when API key is missing."""
        result = utils.fetch_omdb_details(None, imdb_id='tt0133093')
        assert result is None

        result = utils.fetch_omdb_details('', imdb_id='tt0133093')
        assert result is None

        result = utils.fetch_omdb_details('your_omdb_api_key', imdb_id='tt0133093')
        assert result is None

    def test_fetch_no_title_or_id(self):
        """Test that function returns None when neither title nor ID is provided."""
        result = utils.fetch_omdb_details('test_api_key')
        assert result is None


class TestGeminiIntegration:
    """Tests for Gemini API integration functions."""

    def test_configure_gemini_success(self):
        """Test successful Gemini configuration."""
        with patch('utils_faiss.genai.configure') as mock_configure, \
             patch('utils_faiss.genai.GenerativeModel') as mock_model:

            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance

            model = utils._configure_gemini('test_api_key')

            mock_configure.assert_called_once_with(api_key='test_api_key')
            mock_model.assert_called_once_with('gemini-1.5-flash')
            assert model == mock_model_instance

    def test_configure_gemini_no_api_key(self):
        """Test that configuration fails gracefully without API key."""
        model = utils._configure_gemini(None)
        assert model is None

        model = utils._configure_gemini('')
        assert model is None

    def test_configure_gemini_exception(self):
        """Test handling of configuration exceptions."""
        with patch('utils_faiss.genai.configure', side_effect=Exception('Config error')):
            model = utils._configure_gemini('test_api_key')
            assert model is None

    def test_generate_gemini_summary_success(self, mock_gemini_summary_response):
        """Test successful summary generation."""
        with patch('utils_faiss._configure_gemini') as mock_config:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = mock_gemini_summary_response
            mock_model.generate_content.return_value = mock_response
            mock_config.return_value = mock_model

            plot = "A computer hacker learns about the matrix..."
            summary = utils.generate_gemini_summary('test_api_key', plot, 'The Matrix')

            assert summary == mock_gemini_summary_response
            mock_model.generate_content.assert_called_once()

    def test_generate_gemini_summary_no_api_key(self):
        """Test summary generation without API key."""
        summary = utils.generate_gemini_summary(None, 'Some plot', 'Movie Title')
        assert 'missing plot or API key' in summary.lower()

    def test_generate_gemini_summary_no_plot(self):
        """Test summary generation without plot."""
        summary = utils.generate_gemini_summary('test_api_key', None, 'Movie Title')
        assert 'missing plot or API key' in summary.lower()

        summary = utils.generate_gemini_summary('test_api_key', '', 'Movie Title')
        assert 'missing plot or API key' in summary.lower()

        summary = utils.generate_gemini_summary('test_api_key', 'N/A', 'Movie Title')
        assert 'missing plot or API key' in summary.lower()

    def test_generate_gemini_summary_empty_response(self):
        """Test handling of empty response from Gemini."""
        with patch('utils_faiss._configure_gemini') as mock_config:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "   "  # Whitespace only
            mock_model.generate_content.return_value = mock_response
            mock_config.return_value = mock_model

            summary = utils.generate_gemini_summary('test_api_key', 'Plot', 'Title')
            assert "didn't return a summary" in summary.lower()

    def test_get_gemini_recommendations_success(self):
        """Test successful recommendation retrieval from Gemini."""
        with patch('utils_faiss._configure_gemini') as mock_config:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "1. The Matrix\n2. Blade Runner\n3. Inception\n4. Total Recall\n5. Dark City"
            mock_model.generate_content.return_value = mock_response
            mock_config.return_value = mock_model

            recs = utils.get_gemini_recommendations(
                'test_api_key',
                'I want a sci-fi movie',
                'Year: 1990-2010'
            )

            assert len(recs) == 5
            assert 'The Matrix' in recs
            assert 'Blade Runner' in recs

    def test_get_gemini_recommendations_no_api_key(self):
        """Test recommendations without API key."""
        recs = utils.get_gemini_recommendations(None, 'Some prompt', 'Filters')
        assert isinstance(recs, list)
        assert len(recs) == 0

    def test_get_gemini_recommendations_parsing(self):
        """Test parsing of various response formats."""
        with patch('utils_faiss._configure_gemini') as mock_config:
            mock_model = Mock()
            mock_response = Mock()
            # Test format without numbers
            mock_response.text = "The Matrix\nBlade Runner\nInception"
            mock_model.generate_content.return_value = mock_response
            mock_config.return_value = mock_model

            recs = utils.get_gemini_recommendations('test_api_key', 'prompt', 'filters')

            assert len(recs) >= 3
            assert 'The Matrix' in recs


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_get_recommendations_with_small_dataset(self):
        """Test recommendations when dataset is very small."""
        # Create a minimal model_data with just 2 movies
        small_df = pd.DataFrame({
            'tconst': ['tt0001', 'tt0002'],
            'primaryTitle': ['Movie 1', 'Movie 2'],
            'startYear': [2000, 2001],
            'genres': ['Action', 'Drama'],
            'averageRating': [7.0, 8.0],
            'numVotes': [1000, 2000],
            'runtimeMinutes': [120, 110]
        })

        # Create minimal FAISS index
        np.random.seed(42)
        vectors = np.random.randn(2, 10).astype('float32')
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        index = faiss.IndexFlatIP(10)
        index.add(vectors)

        model_data = {
            'movie_metadata': small_df,
            'faiss_index': index,
            'id_to_index': pd.Series([0, 1], index=['tt0001', 'tt0002']),
            'index_to_id': pd.Series(['tt0001', 'tt0002'])
        }

        recs = utils.get_movie_recommendations('tt0001', model_data, n=5)

        # Should return at most 1 recommendation (the other movie)
        assert len(recs) <= 1

    def test_unicode_handling_in_titles(self, mock_model_data):
        """Test handling of unicode characters in movie titles."""
        # Add a movie with unicode characters
        unicode_movie = pd.DataFrame({
            'tconst': ['tt8888888'],
            'primaryTitle': ['Amélie'],
            'startYear': [2001],
            'genres': ['Romance,Comedy'],
            'averageRating': [8.3],
            'numVotes': [730000],
            'runtimeMinutes': [122]
        })

        mock_model_data['movie_metadata'] = pd.concat([
            mock_model_data['movie_metadata'],
            unicode_movie
        ])
        mock_model_data['title_to_ids']['amélie'] = ['tt8888888']

        result = utils.get_movie_by_title('Amélie', mock_model_data)
        assert not result.empty
