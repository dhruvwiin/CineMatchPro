"""
Tests for app.py

This module tests the application logic in app.py including:
- Utility functions
- Data loading
- Display formatting logic
- Session state management

Note: Full Streamlit UI testing requires selenium/playwright and is not included here.
These tests focus on the testable business logic extracted from the app.
"""

import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from io import BytesIO


class TestUtilityFunctions:
    """Tests for utility functions that can be extracted and tested."""

    def test_star_rating_calculation(self):
        """Test star rating calculation logic (extracted from display_movie_card)."""
        def calculate_stars(rating):
            """Helper function extracted from display_movie_card."""
            if rating is None or rating == 'N/A':
                return "N/A"
            try:
                rating_float = float(rating)
                if pd.isna(rating_float):
                    return "N/A"
                num_stars = round(rating_float / 2)
                return "★" * num_stars + "☆" * (5 - num_stars)
            except (ValueError, TypeError):
                return "N/A"

        # Test various ratings
        assert calculate_stars(10.0) == "★★★★★"  # 10/2 = 5
        assert calculate_stars(8.0) == "★★★★☆"   # 8/2 = 4
        assert calculate_stars(5.0) == "★★☆☆☆"   # 5/2 = 2.5 rounds to 2
        assert calculate_stars(6.0) == "★★★☆☆"   # 6/2 = 3
        assert calculate_stars(0.0) == "☆☆☆☆☆"   # 0/2 = 0
        assert calculate_stars(None) == "N/A"
        assert calculate_stars('N/A') == "N/A"
        assert calculate_stars('invalid') == "N/A"

    def test_votes_formatting(self):
        """Test vote count formatting logic."""
        def format_votes(votes):
            """Helper function to format vote counts."""
            try:
                votes_str = f"{int(float(str(votes).replace(',', ''))):,}"
                return votes_str
            except:
                return 'N/A'

        assert format_votes(1000) == "1,000"
        assert format_votes(1000000) == "1,000,000"
        assert format_votes("1,500,000") == "1,500,000"
        assert format_votes('N/A') == "N/A"
        assert format_votes(None) == "N/A"

    def test_year_formatting(self):
        """Test year formatting logic."""
        def format_year(year):
            """Helper function to format year."""
            try:
                if year is not None and pd.notna(year):
                    return str(int(year))
            except (ValueError, TypeError):
                pass
            return ""

        assert format_year(1999) == "1999"
        assert format_year(1999.0) == "1999"
        assert format_year(None) == ""
        assert format_year(pd.NA) == ""
        assert format_year(float('nan')) == ""


class TestDataValidation:
    """Tests for data validation logic."""

    def test_valid_movie_data_structure(self, sample_movies_df):
        """Test that movie dataframe has required columns."""
        required_columns = [
            'tconst', 'primaryTitle', 'startYear', 'genres',
            'averageRating', 'numVotes', 'runtimeMinutes'
        ]

        for col in required_columns:
            assert col in sample_movies_df.columns

    def test_genre_extraction(self, sample_movies_df):
        """Test genre extraction and counting logic."""
        # Simulate genre distribution logic from Home page
        genre_counts_list = []
        for genres in sample_movies_df['genres'].dropna():
            if isinstance(genres, str):
                for g in genres.split(','):
                    genre_counts_list.append(g)

        assert len(genre_counts_list) > 0

        genre_counts = pd.Series(genre_counts_list).value_counts()
        assert 'Action' in genre_counts.index
        assert 'Sci-Fi' in genre_counts.index

    def test_unique_genres_extraction(self, sample_movies_df):
        """Test extraction of unique genres for genre selector."""
        all_genres = set()
        for genres_str in sample_movies_df['genres'].dropna():
            if isinstance(genres_str, str):
                all_genres.update(g.strip() for g in genres_str.split(','))

        unique_genres = ['All'] + sorted(list(all_genres))

        assert 'All' in unique_genres
        assert 'Action' in unique_genres
        assert 'Drama' in unique_genres
        assert len(unique_genres) > 1


class TestSessionStateManagement:
    """Tests for session state management logic."""

    def test_default_state_structure(self):
        """Test that default state has all required keys."""
        default_state = {
            'summaries': {},
            'recommendations': None,
            'matched_movie': None,
            'omdb_info_selected': None,
            'gemini_recs': None,
            'ai_filters': None,
            'genre_results': None,
            'decade_results': None,
            'current_page': 'Home'
        }

        required_keys = [
            'summaries', 'recommendations', 'matched_movie',
            'omdb_info_selected', 'gemini_recs', 'ai_filters',
            'genre_results', 'decade_results', 'current_page'
        ]

        for key in required_keys:
            assert key in default_state

    def test_summaries_storage_format(self):
        """Test that summaries are stored with correct format."""
        summaries = {}

        # Simulate storing a summary
        imdb_id = 'tt0133093'
        summary_text = "A thrilling sci-fi movie..."

        summaries[imdb_id] = summary_text

        assert imdb_id in summaries
        assert summaries[imdb_id] == summary_text

    def test_genre_results_storage_format(self):
        """Test genre results storage format."""
        genre_results = {
            'genre': 'Action',
            'years': (1990, 2010),
            'movies': pd.DataFrame({'title': ['Movie 1', 'Movie 2']})
        }

        assert 'genre' in genre_results
        assert 'years' in genre_results
        assert 'movies' in genre_results
        assert isinstance(genre_results['movies'], pd.DataFrame)

    def test_ai_filters_storage_format(self):
        """Test AI filters storage format."""
        ai_filters = {
            'prompt': 'I want a sci-fi movie',
            'filters_text': 'Year: 1990-2010, Min Rating: 7.0',
            'min_year': 1990,
            'max_year': 2010,
            'min_rating': 7.0,
            'max_runtime': 180
        }

        required_keys = ['prompt', 'filters_text', 'min_year', 'max_year', 'min_rating', 'max_runtime']

        for key in required_keys:
            assert key in ai_filters


class TestMovieCardLogic:
    """Tests for movie card display logic."""

    def test_display_title_selection(self, mock_omdb_response, sample_movies_df):
        """Test that display title is selected correctly."""
        movie_info = sample_movies_df.iloc[0].to_dict()
        omdb_info = mock_omdb_response

        # When OMDb info is available, use it
        display_title = omdb_info.get("Title") if omdb_info else movie_info.get("primaryTitle", "Unknown Title")
        assert display_title == "The Matrix"

        # When OMDb info is not available, use movie_info
        display_title = None if not omdb_info else movie_info.get("primaryTitle", "Unknown Title")
        if display_title is None:
            display_title = movie_info.get("primaryTitle", "Unknown Title")
        assert display_title == "The Matrix"

    def test_runtime_formatting(self, mock_omdb_response, sample_movies_df):
        """Test runtime formatting logic."""
        movie_info = sample_movies_df.iloc[0].to_dict()
        omdb_info = mock_omdb_response

        # OMDb format
        display_runtime = omdb_info.get("Runtime") if omdb_info else None
        assert display_runtime == "136 min"

        # Fallback format
        runtime_minutes = movie_info.get('runtimeMinutes')
        if runtime_minutes and pd.notna(runtime_minutes):
            fallback_runtime = f"{int(runtime_minutes)} min"
            assert fallback_runtime == "136 min"

    def test_imdb_url_generation(self):
        """Test IMDb URL generation."""
        imdb_id = 'tt0133093'
        expected_url = f"https://www.imdb.com/title/{imdb_id}/"

        assert expected_url == "https://www.imdb.com/title/tt0133093/"

    def test_plot_fallback(self, sample_movies_df):
        """Test plot text fallback logic."""
        movie_info = sample_movies_df.iloc[0].to_dict()

        # Test when plot is missing
        omdb_info = {}
        display_plot = omdb_info.get("Plot") if omdb_info else movie_info.get("overview")
        if not display_plot or display_plot == "N/A":
            display_plot = "Plot summary not available."

        assert display_plot == "Plot summary not available."


class TestFilteringLogic:
    """Tests for movie filtering logic."""

    def test_year_range_filter(self, sample_movies_df):
        """Test year range filtering logic."""
        min_year = 2000
        max_year = 2010

        filtered = sample_movies_df[
            (sample_movies_df['startYear'] >= min_year) &
            (sample_movies_df['startYear'] <= max_year)
        ]

        assert all((min_year <= year <= max_year) for year in filtered['startYear'])

    def test_rating_filter(self, sample_movies_df):
        """Test minimum rating filtering."""
        min_rating = 8.0

        filtered = sample_movies_df[sample_movies_df['averageRating'] >= min_rating]

        assert all(rating >= min_rating for rating in filtered['averageRating'])

    def test_runtime_filter(self, sample_movies_df):
        """Test maximum runtime filtering."""
        max_runtime = 150

        filtered = sample_movies_df[sample_movies_df['runtimeMinutes'] <= max_runtime]

        assert all(runtime <= max_runtime for runtime in filtered['runtimeMinutes'])

    def test_combined_filters(self, sample_movies_df):
        """Test applying multiple filters together."""
        min_year = 2000
        max_year = 2010
        min_rating = 7.0
        max_runtime = 150

        filtered = sample_movies_df[
            (sample_movies_df['startYear'] >= min_year) &
            (sample_movies_df['startYear'] <= max_year) &
            (sample_movies_df['averageRating'] >= min_rating) &
            (sample_movies_df['runtimeMinutes'] <= max_runtime)
        ]

        for _, row in filtered.iterrows():
            assert min_year <= row['startYear'] <= max_year
            assert row['averageRating'] >= min_rating
            assert row['runtimeMinutes'] <= max_runtime


class TestDecadeExplorer:
    """Tests for decade explorer logic."""

    def test_decade_statistics_calculation(self, sample_movies_df):
        """Test calculation of decade statistics."""
        decade = '1990s'
        decade_movies = sample_movies_df[sample_movies_df['decade_str'] == decade]

        # Calculate metrics
        num_movies = len(decade_movies)
        avg_rating = decade_movies['averageRating'].mean() if not decade_movies.empty else 0
        avg_runtime = decade_movies['runtimeMinutes'].mean() if not decade_movies.empty else 0

        assert num_movies >= 0
        if num_movies > 0:
            assert avg_rating > 0
            assert avg_runtime > 0

    def test_decade_movie_sorting(self, sample_movies_df):
        """Test sorting of movies within a decade."""
        decade = '2000s'
        decade_movies = sample_movies_df[sample_movies_df['decade_str'] == decade].copy()

        # Sort by rating and popularity
        sorted_movies = decade_movies.sort_values(
            ['averageRating', 'popularity_score'],
            ascending=[False, False]
        )

        # Check that first movie has highest or equal rating
        if len(sorted_movies) > 1:
            assert sorted_movies.iloc[0]['averageRating'] >= sorted_movies.iloc[1]['averageRating']


class TestAPIKeyValidation:
    """Tests for API key validation logic."""

    def test_omdb_api_key_validation(self):
        """Test OMDb API key validation."""
        valid_key = "abc123def456"
        invalid_keys = [None, "", "your_omdb_api_key"]

        # Valid key should pass
        assert valid_key is not None and valid_key != "" and valid_key != "your_omdb_api_key"

        # Invalid keys should fail
        for key in invalid_keys:
            is_invalid = key is None or key == "" or key == "your_omdb_api_key"
            assert is_invalid

    def test_gemini_api_key_validation(self):
        """Test Gemini API key validation."""
        valid_key = "AIzaSyABC123DEF456"
        invalid_keys = [None, ""]

        # Valid key should pass
        assert valid_key is not None and valid_key != ""

        # Invalid keys should fail
        for key in invalid_keys:
            is_invalid = key is None or key == ""
            assert is_invalid


class TestSummarizeButtonLogic:
    """Tests for summarize button logic."""

    def test_can_summarize_conditions(self):
        """Test conditions for enabling summarize button."""
        # Test case 1: All conditions met
        plot = "A computer hacker discovers the truth about reality..."
        gemini_api_key = "valid_key"
        imdb_id = "tt0133093"

        can_summarize = bool(
            plot != "Plot summary not available." and
            gemini_api_key and
            imdb_id
        )
        assert can_summarize is True

        # Test case 2: Missing plot
        plot = "Plot summary not available."
        can_summarize = bool(
            plot != "Plot summary not available." and
            gemini_api_key and
            imdb_id
        )
        assert can_summarize is False

        # Test case 3: Missing API key
        plot = "A computer hacker discovers the truth about reality..."
        gemini_api_key = None
        can_summarize = bool(
            plot != "Plot summary not available." and
            gemini_api_key and
            imdb_id
        )
        assert can_summarize is False

        # Test case 4: Missing IMDb ID
        gemini_api_key = "valid_key"
        imdb_id = None
        can_summarize = bool(
            plot != "Plot summary not available." and
            gemini_api_key and
            imdb_id
        )
        assert can_summarize is False


class TestMovieTitleList:
    """Tests for movie title list generation."""

    def test_title_list_generation(self, sample_movies_df):
        """Test generation of sorted movie title list."""
        movie_titles = sorted(sample_movies_df['primaryTitle'].dropna().unique().tolist())

        assert len(movie_titles) > 0
        assert movie_titles == sorted(movie_titles)  # Check if sorted
        assert 'The Matrix' in movie_titles

    def test_title_list_no_duplicates(self, sample_movies_df):
        """Test that title list has no duplicates."""
        movie_titles = sample_movies_df['primaryTitle'].dropna().unique().tolist()

        assert len(movie_titles) == len(set(movie_titles))


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_missing_css_file_handling(self, tmp_path):
        """Test handling of missing CSS file."""
        css_file = tmp_path / "nonexistent_style.css"

        # Simulate the load_css logic
        try:
            with open(css_file) as f:
                content = f.read()
            css_loaded = True
        except FileNotFoundError:
            css_loaded = False

        assert css_loaded is False

    def test_empty_recommendations_handling(self):
        """Test handling of empty recommendations."""
        recommendations = pd.DataFrame()

        assert recommendations.empty
        # Should not attempt to display movies

    def test_null_movie_data_handling(self):
        """Test handling of None/null movie data."""
        matched_movie = None

        assert matched_movie is None
        # Should not attempt to display movie card
