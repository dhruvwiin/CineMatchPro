import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pickle
import os
import gc
import faiss

'''
This function takes the cleaned movie data and builds all the files
needed for the recommendation system. It creates things like the
feature transformer, the scaler, and the special Faiss index for fast searching.
It reads from the input_file (like data/imdb_final_cleaned.csv) and saves
everything into the output_dir (like model_faiss/).
'''
def build_recommendation_model(input_file, output_dir='model_faiss'):
    print(f"Building Faiss-based recommendation model from {input_file}...")
    print(f"Output directory: {output_dir}")

    '''
    Make the folder to save our model files if it doesn't already exist.
    Usually this will be the 'model_faiss' folder.
    '''
    os.makedirs(output_dir, exist_ok=True)

    '''
    Load the big CSV file with all the movie info.
    Make sure the input_file path is correct!
    '''
    print("Loading dataset...")
    try:
        if not os.path.exists(input_file):
             raise FileNotFoundError(f"Input file not found at specified path: {input_file}")
        df = pd.read_csv(input_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the path is correct relative to where you are running the script.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


    print(f"Processing {len(df)} movies.")

    '''
    Get the movie features ready for the model.
    '''

    '''
    Make sure every movie has something in the 'genres' column,
    even if it's just empty text.
    '''
    df['genres'] = df['genres'].fillna('')

    '''
    Handle the number features like year, rating, votes, etc.
    Fill in missing numbers with the average (median).
    Then scale them so big numbers don't overpower small ones.
    '''
    numerical_cols = [
        'startYear', 'runtimeMinutes', 'averageRating', 'numVotes',
        'director_movie_count', 'writer_movie_count',
        'cast1_movie_count', 'cast2_movie_count', 'cast3_movie_count',
        'popularity_score'
    ]
    numerical_cols = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    print(f"Using {len(numerical_cols)} numerical features: {numerical_cols}")
    for col in numerical_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    scaler = StandardScaler()
    if not df[numerical_cols].empty:
        scaled_numerical_features = scaler.fit_transform(df[numerical_cols].astype(float))
        print("Numerical features scaled.")
    else:
        scaled_numerical_features = np.array([]).reshape(len(df), 0)

    '''
    Handle the text features, mainly the genres.
    Use TF-IDF to turn the genre words into numbers that represent
    how important each genre is for each movie.
    '''
    print("Applying TF-IDF to genres...")
    df['genres_processed'] = df['genres'].str.replace(',', ' ')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.7)
    genre_tfidf_features = tfidf_vectorizer.fit_transform(df['genres_processed'])
    print(f"TF-IDF applied. Shape: {genre_tfidf_features.shape}")

    '''
    Stick the processed number features and text features together
    into one big table (matrix).
    '''
    print("Combining features...")
    combined_features_sparse = hstack([genre_tfidf_features, scaled_numerical_features]).tocsr()
    print(f"Combined sparse feature matrix shape: {combined_features_sparse.shape}")

    '''
    Get the combined features ready for Faiss.
    Faiss likes a specific format (numpy array of float32).
    Converting can use a lot of memory if you have tons of movies/features!
    Also normalize the features, which helps Faiss find similar movies correctly.
    '''
    print("Preparing data for Faiss index...")
    try:
        combined_features_dense = combined_features_sparse.astype(np.float32).toarray()
        print(f"Converted to dense matrix. Shape: {combined_features_dense.shape}")
    except MemoryError:
        print("\nError: Memory Error converting sparse features to dense matrix for Faiss.")
        print(f"The shape ({combined_features_sparse.shape}) might be too large for available RAM.")
        print("Consider reducing data size (sampling/filtering) or exploring Faiss indexes that handle sparse data directly (more advanced).")
        return None
    except Exception as e:
        print(f"\nError converting features to dense matrix: {e}")
        return None

    print("Normalizing features (L2 norm)...")
    normalize(combined_features_dense, norm='l2', axis=1, copy=False)

    '''
    Clean up variables we don't need anymore to save memory.
    '''
    del combined_features_sparse
    del scaled_numerical_features
    del genre_tfidf_features
    gc.collect()

    '''
    Build the Faiss index. This is like a super-fast search structure.
    We use IndexFlatIP which works well for finding similar items based
    on their features (after we normalized them).
    Then we add all our movie feature vectors into this index.
    '''
    print("Building Faiss index...")
    dimension = combined_features_dense.shape[1]
    index = faiss.IndexFlatIP(dimension)
    try:
        index.add(combined_features_dense)
        print(f"Faiss index built. Total vectors indexed: {index.ntotal}")
    except Exception as e:
         print(f"\nError adding vectors to Faiss index: {e}")
         return None

    '''
    Clean up the big dense matrix now that it's in the Faiss index.
    '''
    del combined_features_dense
    gc.collect()

    '''
    Save all the important things we created into files
    inside the output_dir (e.g., 'model_faiss').
    This includes the Faiss index itself, the scaler, the TF-IDF thing,
    and some helper files for mapping IDs and titles.
    '''
    print("Saving model artifacts...")

    '''
    Save the Faiss index to a file named faiss_index.idx.
    '''
    index_path = os.path.join(output_dir, 'faiss_index.idx')
    try:
        faiss.write_index(index, index_path)
        print(f"Saved Faiss index to {index_path}")
    except Exception as e:
        print(f"\nError saving Faiss index: {e}")
        pass

    '''
    Save the scaler object (used for number features).
    '''
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")

    '''
    Save the TF-IDF vectorizer object (used for genre features).
    '''
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"Saved vectorizer to {vectorizer_path}")

    '''
    Save a smaller CSV file containing just the movie info we need for the app.
    Make sure it includes 'tconst' (the movie ID).
    The order of movies in this file MUST match the order they were added to Faiss.
    '''
    metadata_cols = ['tconst', 'primaryTitle', 'originalTitle', 'startYear',
                     'genres', 'averageRating', 'numVotes', 'runtimeMinutes',
                     'popularity_score', 'decade_str']
    metadata_cols = [col for col in metadata_cols if col in df.columns]
    movie_metadata = df[metadata_cols].copy()

    '''
    Create and save mappings so we can easily find a movie's Faiss index
    number from its ID ('tconst'), and vice-versa. This is important!
    '''
    if 'tconst' not in movie_metadata.columns:
        print("\nError: 'tconst' column missing from final DataFrame. Cannot create index_to_id mapping.")
        return None

    # Assumes the row number (0, 1, 2...) matches the Faiss index position
    index_to_id = pd.Series(movie_metadata['tconst'].values) # Faiss index -> tconst
    id_to_index = pd.Series(index_to_id.index, index=index_to_id.values) # tconst -> Faiss index

    metadata_path = os.path.join(output_dir, 'movie_metadata.csv')
    movie_metadata.to_csv(metadata_path, index=False)
    print(f"Saved movie metadata to {metadata_path} (order matches Faiss index)")

    id_index_path = os.path.join(output_dir, 'id_to_index.pkl')
    with open(id_index_path, 'wb') as f:
        pickle.dump(id_to_index, f)
    index_id_path = os.path.join(output_dir, 'index_to_id.pkl')
    with open(index_id_path, 'wb') as f:
        pickle.dump(index_to_id, f)

    '''
    Save a mapping from movie titles (lowercase) to their IDs ('tconst').
    Needed for searching by title in the app.
    '''
    title_to_ids = df.groupby(df['primaryTitle'].str.lower())['tconst'].apply(list).to_dict()
    title_map_path = os.path.join(output_dir, 'title_to_ids.pkl')
    with open(title_map_path, 'wb') as f:
        pickle.dump(title_to_ids, f)
    print(f"Saved title/ID mappings to {title_map_path}")

    '''
    Optionally save the names of all the features used in the model.
    '''
    try:
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
        all_feature_names = tfidf_feature_names + numerical_cols
        features_path = os.path.join(output_dir, 'feature_names.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(all_feature_names, f)
        print(f"Saved feature names to {features_path}")
    except Exception as e:
        print(f"Warning: Could not save feature names. Error: {e}")


    print(f"\nModel build complete. Artifacts saved to {output_dir}")
    print(f"Total movies processed and indexed: {len(df)}")
    print(f"Faiss index contains {index.ntotal if 'index' in locals() else 'N/A'} vectors.")

    return output_dir

'''
This part only runs if you execute this script directly
(e.g., by typing 'python model.py' in the terminal).
It sets up the input file path ('data/imdb_final_cleaned.csv')
and the output directory path ('model_faiss/'), then calls
the build_recommendation_model function to do the work.
'''
if __name__ == "__main__":
    # Define where the input data is and where the model files should go
    INPUT_CSV_PATH = os.path.join('data', 'imdb_final_cleaned.csv')
    OUTPUT_MODEL_DIR = 'model_faiss'

    # Figure out the full paths based on where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir

    absolute_input_path = os.path.join(project_root, INPUT_CSV_PATH)
    absolute_output_dir = os.path.join(project_root, OUTPUT_MODEL_DIR)

    print(f"Project Root (inferred): {project_root}")
    print(f"Absolute Input Path: {absolute_input_path}")
    print(f"Absolute Output Dir: {absolute_output_dir}")

    # Start building the model!
    build_recommendation_model(absolute_input_path, absolute_output_dir)