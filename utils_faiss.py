import pandas as pd
import numpy as np
import pickle
import os
import requests
import faiss
import gc
import google.generativeai as genai # For using the Google AI (Gemini)
from google.api_core import exceptions as google_exceptions # For handling specific Google API errors

'''
This sets the default folder name where we expect to find the model files
like the index and mappings. Usually 'model_faiss'.
'''
DEFAULT_MODEL_DIR = 'model_faiss'

'''
This function loads all the files that model.py created.
It reads the Faiss index, the movie data CSV, and the mapping files (pickle files)
from the specified model_dir folder (usually 'model_faiss').
It puts everything into a dictionary called model_data.
'''
def load_model_data(model_dir=DEFAULT_MODEL_DIR):
    print(f"Loading model data from: {model_dir}")
    model_data = {}
    try:
        '''Load the main movie info csv'''
        metadata_path = os.path.join(model_dir, 'movie_metadata.csv')
        if not os.path.exists(metadata_path): raise FileNotFoundError("movie_metadata.csv not found.")
        model_data['movie_metadata'] = pd.read_csv(metadata_path)
        print("Loaded metadata.")

        '''Load the faiss index file (the fast search thing)'''
        index_path = os.path.join(model_dir, 'faiss_index.idx')
        if not os.path.exists(index_path): raise FileNotFoundError("faiss_index.idx not found.")
        model_data['faiss_index'] = faiss.read_index(index_path)
        print(f"Loaded Faiss index. Index contains {model_data['faiss_index'].ntotal} vectors.")

        '''Load the helper mapping files'''
        id_index_path = os.path.join(model_dir, 'id_to_index.pkl')
        index_id_path = os.path.join(model_dir, 'index_to_id.pkl')
        title_map_path = os.path.join(model_dir, 'title_to_ids.pkl')
        if not os.path.exists(id_index_path): raise FileNotFoundError("id_to_index.pkl not found.")
        if not os.path.exists(index_id_path): raise FileNotFoundError("index_to_id.pkl not found.")
        if not os.path.exists(title_map_path): raise FileNotFoundError("title_to_ids.pkl not found.")
        with open(id_index_path, 'rb') as f: model_data['id_to_index'] = pickle.load(f)
        with open(index_id_path, 'rb') as f: model_data['index_to_id'] = pickle.load(f)
        with open(title_map_path, 'rb') as f: model_data['title_to_ids'] = pickle.load(f)
        print("Loaded ID/Title mappings.")

        '''Load other optional files if they exist'''
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        if os.path.exists(scaler_path):
             with open(scaler_path, 'rb') as f: model_data['scaler'] = pickle.load(f)
             print("Loaded scaler.")
        if os.path.exists(vectorizer_path):
             with open(vectorizer_path, 'rb') as f: model_data['tfidf_vectorizer'] = pickle.load(f)
             print("Loaded TF-IDF vectorizer.")

    except FileNotFoundError as e:
        print(f"Error loading model data: {e}.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading model data: {e}")
        import traceback; traceback.print_exc()
        return None

    '''Check if the faiss index size matches the mapping size'''
    if 'faiss_index' in model_data and 'index_to_id' in model_data:
        if model_data['faiss_index'].ntotal != len(model_data['index_to_id']):
             print(f"\n*** Warning: Faiss index size ({model_data['faiss_index'].ntotal}) does not match mapping size ({len(model_data['index_to_id'])}). ***")

    print("Model data loading complete.")
    return model_data

'''
Find movies in our data based on the title typed by the user.
Uses the title_to_ids mapping loaded earlier.
Tries exact match first, then partial match.
'''
def get_movie_by_title(title, model_data):
    if not model_data or 'title_to_ids' not in model_data or 'movie_metadata' not in model_data:
        return pd.DataFrame()
    title_lower = title.lower()
    title_to_ids = model_data['title_to_ids']
    metadata = model_data['movie_metadata']
    matching_ids = title_to_ids.get(title_lower, [])
    if not matching_ids:
        try:
            # Simple partial match check
            partial_matches = [ids for t, ids in title_to_ids.items() if title_lower in t]
            if partial_matches: matching_ids = [item for sublist in partial_matches for item in sublist]
        except Exception as e: print(f"Error during partial title match: {e}")
    if matching_ids:
        unique_matching_ids = list(set(matching_ids))
        return metadata[metadata['tconst'].isin(unique_matching_ids)].copy()
    else:
        return pd.DataFrame()

'''
This is the core recommendation function using Faiss.
Given a movie ID (like 'tt0111161'), it finds that movie's vector
in the Faiss index, then searches the index for the 'n' closest vectors (movies).
It uses the id_to_index and index_to_id mappings to convert between
movie IDs and the Faiss index position numbers.
Returns a pandas DataFrame with the recommended movies and similarity scores.
'''
def get_movie_recommendations(movie_id, model_data, n=10):
    if not model_data: return pd.DataFrame()
    required_keys = ['faiss_index', 'id_to_index', 'index_to_id', 'movie_metadata']
    if not all(key in model_data for key in required_keys): return pd.DataFrame()

    faiss_index = model_data['faiss_index']
    id_to_index = model_data['id_to_index']
    index_to_id = model_data['index_to_id']
    metadata = model_data['movie_metadata']

    '''Find the Faiss index number for the given movie ID'''
    if movie_id not in id_to_index: return pd.DataFrame()
    try:
        movie_idx = id_to_index[movie_id]
        if movie_idx < 0 or movie_idx >= faiss_index.ntotal: return pd.DataFrame()
    except Exception as e: return pd.DataFrame()

    '''Get the vector for our movie and search Faiss for similar ones'''
    try:
        movie_idx_int = int(movie_idx)
        # Need to get the actual vector data for the movie from the index
        if not hasattr(faiss_index, 'reconstruct'): return pd.DataFrame() # Check if index supports reconstruct
        query_vector = faiss_index.reconstruct(movie_idx_int).reshape(1, -1)
        # Search the index: find n+1 neighbors (because one will be the movie itself)
        distances, neighbor_indices = faiss_index.search(query_vector, n + 1)
        distances = distances.flatten()
        neighbor_indices = neighbor_indices.flatten()
    except Exception as e:
        print(f"Error during Faiss search for movie index {movie_idx}: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

    '''Process the search results'''
    recommendations_data = []
    valid_neighbors_found = 0
    for i, idx in enumerate(neighbor_indices):
        # Skip invalid indices or the movie itself
        if idx < 0 or idx == movie_idx_int or idx >= len(index_to_id): continue
        try:
            # Get the movie ID ('tconst') from the Faiss index number
            neighbor_movie_id = index_to_id[idx]
            # The distance is actually the similarity score here (since we used IndexFlatIP on normalized vectors)
            similarity_score = distances[i]
            recommendations_data.append({'tconst': neighbor_movie_id, 'similarity': similarity_score})
            valid_neighbors_found += 1
            if valid_neighbors_found >= n: break # Stop once we have enough recommendations
        except IndexError: continue # Skip if index is somehow out of bounds for the mapping

    if not recommendations_data: return pd.DataFrame()

    '''Get the full details for the recommended movie IDs'''
    recs_df = pd.DataFrame(recommendations_data)
    recommendations_final = pd.merge(recs_df, metadata, on='tconst', how='left')
    recommendations_final = recommendations_final.sort_values('similarity', ascending=False)
    recommendations_final.fillna({'primaryTitle': 'Unknown Title', 'startYear': 'N/A'}, inplace=True)
    return recommendations_final

'''
Helper function to get recommendations starting from a movie title.
It first finds the movie ID using get_movie_by_title,
then calls get_movie_recommendations with that ID.
Handles cases where a title might match multiple movies (picks the most popular one).
'''
def get_recommendations_by_title(title, model_data, n=10):
    matches_df = get_movie_by_title(title, model_data)
    if matches_df.empty: return pd.DataFrame(), None
    # If multiple matches, pick the one with most votes or just the first alphabetically
    if len(matches_df) > 1:
        if 'numVotes' in matches_df.columns and pd.api.types.is_numeric_dtype(matches_df['numVotes']):
             matches_df = matches_df.sort_values('numVotes', ascending=False, na_position='last')
        else: matches_df = matches_df.sort_values('primaryTitle', ascending=True)
    movie = matches_df.iloc[0] # Take the best match
    movie_id = movie['tconst']
    recommendations = get_movie_recommendations(movie_id, model_data, n)
    return recommendations, movie

'''
Filters the loaded movie metadata (from movie_metadata.csv)
to find movies matching a specific genre and year range.
Used for the 'Explore by Genre' page.
Sorts results by popularity/rating.
'''
def get_top_movies_by_genre(genre, year_range, model_data, n=20):
    if not model_data or 'movie_metadata' not in model_data: return pd.DataFrame()
    metadata = model_data['movie_metadata']
    try:
        # Make sure year is a number before filtering
        metadata['startYear'] = pd.to_numeric(metadata['startYear'], errors='coerce')
        filtered = metadata[(metadata['startYear'].notna()) & (metadata['startYear'] >= year_range[0]) & (metadata['startYear'] <= year_range[1])].copy()
        # Filter by genre if not 'All'
        if genre != 'All':
            if 'genres' in filtered.columns:
                filtered = filtered[filtered['genres'].astype(str).str.contains(genre, case=False, na=False)]
        # Sort the results
        sort_columns, ascending_order = [], []
        if 'popularity_score' in filtered.columns and pd.api.types.is_numeric_dtype(filtered['popularity_score']):
             sort_columns.append('popularity_score'); ascending_order.append(False) # Higher score first
        if 'averageRating' in filtered.columns and pd.api.types.is_numeric_dtype(filtered['averageRating']):
            sort_columns.append('averageRating'); ascending_order.append(False) # Higher rating first
        if sort_columns: filtered = filtered.sort_values(sort_columns, ascending=ascending_order, na_position='last')
        elif 'primaryTitle' in filtered.columns: filtered = filtered.sort_values('primaryTitle') # Fallback sort
        return filtered.head(n) # Return the top N results
    except Exception as e:
        print(f"Error filtering movies by genre/year: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()

'''
Functions for talking to external APIs
'''

'''OMDb API stuff'''
OMDB_API_URL = "http://www.omdbapi.com/" # Website address for OMDb
'''
Gets extra movie details (like Poster, full Plot, Director, Actors etc.)
from the OMDb API using either the IMDb ID (tconst) or title/year.
Needs an OMDb API key to work.
'''
def fetch_omdb_details(api_key, title=None, year=None, imdb_id=None):
    if not api_key or api_key == "your_omdb_api_key": return None # Don't run if no key
    params = {"apikey": api_key, "plot": "full"} # Basic request parameters
    if imdb_id: params["i"] = imdb_id # Search by ID if provided
    elif title:
        params["t"] = title # Otherwise search by title
        if year:
            try: params["y"] = str(int(year)) # Add year if available
            except (ValueError, TypeError): pass
    else: return None # Need title or ID

    try:
        response = requests.get(OMDB_API_URL, params=params, timeout=10) # Make the web request
        response.raise_for_status() # Check for web errors (like 404 Not Found)
        data = response.json() # Read the JSON response
        return data if data.get("Response") == "True" else None # Return data only if OMDb found the movie
    except requests.exceptions.Timeout: print(f"OMDb request timed out for query {params}."); return None
    except requests.exceptions.RequestException as e: print(f"OMDb request failed for query {params}: {e}"); return None
    except Exception as e: print(f"Unexpected error during OMDb fetch for query {params}: {e}"); return None


'''Gemini API stuff'''

'''
Helper function to set up the connection to Google Gemini.
Needs the API key. We use the 'flash' model which is fast.
'''
def _configure_gemini(api_key):
    if not api_key:
        print("Error: Gemini API key not provided.")
        return None
    try:
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash model - supposed to be quick
        model = genai.GenerativeModel('gemini-1.5-flash')
        return model
    except Exception as e:
        print(f"Error configuring Gemini: {e}")
        return None

'''
Asks the Gemini AI to generate a short, fun summary for a movie.
Takes the API key, the movie's plot, and its title.
Handles basic errors like missing plot or API key problems.
'''
def generate_gemini_summary(api_key, plot, title):
    model = _configure_gemini(api_key) # Setup connection
    if not model or not plot or pd.isna(plot) or plot.strip() == "" or plot == "N/A":
        return "Could not generate summary (missing plot or API key)."

    # The instructions we give to the Gemini AI
    prompt = f"""
    Create a fun, engaging, and very short summary (2-3 sentences maximum) for the movie '{title}'.
    Focus on the core premise and tone, without giving away major spoilers.
    Make it sound exciting or intriguing!

    Here is the original plot description:
    "{plot}"

    Fun Summary:
    """
    try:
        # Send the prompt to Gemini and get the response
        # We can add safety_settings here if needed to block bad content
        response = model.generate_content(prompt)
        summary = response.text.strip() # Get the text part of the response
        return summary if summary else "Gemini didn't return a summary."
    except google_exceptions.PermissionDenied:
         print("Error: Gemini API key is invalid or lacks permissions.")
         return "Error: Invalid Gemini API key."
    except google_exceptions.ResourceExhausted:
         print("Error: Gemini API quota exceeded.")
         return "Error: API quota limit reached. Please try again later."
    except Exception as e:
        # Try to catch cases where Gemini blocks the response for safety reasons
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback') and e.response.prompt_feedback.block_reason:
            reason = e.response.prompt_feedback.block_reason
            print(f"Error: Gemini content generation blocked due to safety settings. Reason: {reason}")
            return f"Could not generate summary due to safety filters ({reason})."
        else:
            # Handle other unexpected errors
            print(f"Error generating Gemini summary: {e}")
            import traceback; traceback.print_exc()
            return "Error generating summary."

'''
Asks the Gemini AI to recommend movie titles based on what the user described.
Takes the API key, the user's description (prompt), and any filters they selected.
Tells Gemini to return *only* a numbered list of titles.
Tries to parse that list. Handles API errors.
'''
def get_gemini_recommendations(api_key, user_prompt, filters_text):
    model = _configure_gemini(api_key) # Setup connection
    if not model:
        return [] # Return empty list if setup fails

    # Instructions for Gemini AI
    prompt = f"""
    Based on the following user request and filters, recommend 5 movie titles.
    Prioritize movies that fit the description well, considering the filters secondarily if needed to find matches.
    Return ONLY a numbered list of the movie titles, each on a new line. Do not include years or any other text.

    User Request: "{user_prompt}"
    Filters Applied: "{filters_text if filters_text else 'None'}"

    Recommended Movie Titles:
    1.
    """
    try:
        # Send prompt to Gemini
        response = model.generate_content(prompt)
        response_text = response.text.strip() # Get the text result

        # Try to clean up the response to get just the movie titles
        recommended_titles = []
        lines = response_text.split('\n') # Split into lines
        for line in lines:
            line = line.strip()
            # Check if the line starts with a number and a period (like "1.")
            if '.' in line:
                parts = line.split('.', 1)
                if len(parts) == 2 and parts[0].isdigit():
                    title = parts[1].strip() # Get the text after "number."
                    if title: # Make sure it's not empty
                         recommended_titles.append(title)
                elif line: # If it has a period but no number, keep it? Maybe Gemini messed up.
                    recommended_titles.append(line)
            elif line: # If no period, just keep the line if it's not empty
                recommended_titles.append(line)

        print(f"DEBUG: Gemini recommended titles raw response:\n{response_text}")
        print(f"DEBUG: Parsed titles: {recommended_titles}")
        return recommended_titles[:5] # Return only the first 5 titles found

    except google_exceptions.PermissionDenied:
         print("Error: Gemini API key is invalid or lacks permissions.")
         return ["Error: Invalid Gemini API key."] # Return error message in a list
    except google_exceptions.ResourceExhausted:
         print("Error: Gemini API quota exceeded.")
         return ["Error: API quota limit reached."]
    except Exception as e:
         # Check for safety blocks
        if hasattr(e, 'response') and hasattr(e.response, 'prompt_feedback') and e.response.prompt_feedback.block_reason:
            reason = e.response.prompt_feedback.block_reason
            print(f"Error: Gemini content generation blocked due to safety settings. Reason: {reason}")
            return [f"Error: Could not get recommendations due to safety filters ({reason})."]
        else:
            # Handle other errors
            print(f"Error getting Gemini recommendations: {e}")
            import traceback; traceback.print_exc()
            return ["Error: Could not get recommendations."]