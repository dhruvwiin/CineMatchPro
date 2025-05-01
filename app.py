import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import utils_faiss as utils # Need this for our helper functions from utils_faiss.py


st.set_page_config( #Setting up the basic look of the Streamlit page, like the title and icon.
    page_title="CineMatch Pro - AI Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)   
    
    


#This part keeps track of stuff while the user clicks around. Like storing the recommendations or summaries so they don't disappear.
#We set default values here if they haven't been set yet.
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
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


#Function to load the style.css file.This makes the app look nicer than the default Streamlit style.
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    try:
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css not found. Using default styling.")

load_css()


#Get the secret API keys for OMDb (movie details) and Gemini (AI stuff). These are stored in Streamlit's secrets manager.
#Also loads the main movie data using a function from utils_faiss.py. It expects the model files to be in the 'model_faiss' folder.
OMDB_API_KEY = st.secrets.get("omdb_api", None)
GEMINI_API_KEY = st.secrets.get("gemini_api_key", None)

if not OMDB_API_KEY:
    st.warning("OMDb API key not found. Poster fetching and detailed info will be disabled.")
if not GEMINI_API_KEY:
    st.warning("Gemini API key not found. AI Summarization and AI Movie Finder features will be disabled.")

@st.cache_resource
def load_data():
    '''
    Loads all the necessary model files (index, metadata, etc.)
    Uses the load_model_data function from utils_faiss.py.
    Uses caching so it doesn't reload every time.
    '''
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_faiss")
    model_data = utils.load_model_data(model_dir=model_dir)
    if model_data is None:
        st.error(f"Failed to load model data from '{model_dir}'")
        st.stop()
    return model_data

model_data = load_data()
metadata = model_data['movie_metadata'] if model_data and 'movie_metadata' in model_data else None
if metadata is None:
     st.error("Fatal Error: Movie metadata could not be loaded.")
     st.stop()


#Functions to talk to the external APIs (OMDb and Gemini). We use caching for OMDb so we don't ask for the same movie details too often.
#These call the actual API functions defined in our utils_faiss.py.

@st.cache_data(ttl=3600*24)
def fetch_omdb_details_cached(api_key, title=None, year=None, imdb_id=None):
    # This calls the OMDb function in utils_faiss.py
    return utils.fetch_omdb_details(api_key=api_key, title=title, year=year, imdb_id=imdb_id)

def generate_summary_wrapper(api_key, plot, title):
    # This calls the Gemini summary function in utils_faiss.py
    if not api_key: return "Error: Gemini API key not configured."
    print(f"--- Calling Gemini API to summarize '{title}' ---")
    summary = utils.generate_gemini_summary(api_key=api_key, plot=plot, title=title)
    print(f"--- Gemini summary result for '{title}': {summary} ---")
    return summary

def get_ai_recs_wrapper(api_key, user_prompt, filters_text):
    # This calls the Gemini recommendations function in utils_faiss.py
    if not api_key: return ["Error: Gemini API key not configured."]
    print(f"--- Calling Gemini API for AI recommendations. Prompt: '{user_prompt}', Filters: '{filters_text}' ---")
    recs = utils.get_gemini_recommendations(api_key=api_key, user_prompt=user_prompt, filters_text=filters_text)
    print(f"--- Gemini AI recommendations result: {recs} ---")
    return recs

#This function runs when the 'Summarize Plot' button is clicked. 
#It gets the summary using the Gemini API (via the wrapper)and saves it in the session state.

def handle_summarize_click(imdb_id_arg, plot_arg, title_arg):
    print(f"DEBUG: handle_summarize_click called for imdb_id: {imdb_id_arg}")
    if not GEMINI_API_KEY:
        st.session_state.summaries[imdb_id_arg] = "Error: Gemini API key missing."
        return
    if not plot_arg or plot_arg == "Plot summary not available.":
         st.session_state.summaries[imdb_id_arg] = "Summary impossible: Plot is missing."
         return

    st.session_state.summaries[imdb_id_arg] = "Generating..."
    summary = generate_summary_wrapper(api_key=GEMINI_API_KEY, plot=plot_arg, title=title_arg)
    st.session_state.summaries[imdb_id_arg] = summary
    print(f"DEBUG: Summary stored in session_state for {imdb_id_arg}")


#This function takes movie info and displays it in a nice card format.
#It fetches the poster from OMDb if possible.
#It also includes the button to generate the AI summary.
def display_movie_card(movie_info, omdb_info=None):
    display_title = omdb_info.get("Title") if omdb_info else movie_info.get("primaryTitle", "Unknown Title")
    display_year = omdb_info.get("Year") if omdb_info else movie_info.get("startYear")
    display_year_str = str(int(display_year)) if display_year and pd.notna(display_year) else ""
    display_rating = omdb_info.get("imdbRating", "N/A") if omdb_info else movie_info.get("averageRating", "N/A")
    display_votes = omdb_info.get("imdbVotes", "N/A") if omdb_info else movie_info.get("numVotes", "N/A")
    display_runtime = omdb_info.get("Runtime") if omdb_info else f"{int(movie_info.get('runtimeMinutes', 0))} min" if pd.notna(movie_info.get('runtimeMinutes')) else "N/A"
    display_genre = omdb_info.get("Genre") if omdb_info else movie_info.get("genres", "")
    display_plot = omdb_info.get("Plot") if omdb_info else movie_info.get("overview")
    if not display_plot or display_plot == "N/A": display_plot = "Plot summary not available."
    poster_url = omdb_info.get("Poster", "N/A") if omdb_info else "N/A"
    imdb_id = omdb_info.get("imdbID") if omdb_info else movie_info.get("tconst")
    imdb_url = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
    language = omdb_info.get("Language") if omdb_info else None
    country = omdb_info.get("Country") if omdb_info else None

    stars = ""
    try:
        if display_rating is not None and display_rating != 'N/A':
            rating_float = float(display_rating)
            if not pd.isna(rating_float):
                num_stars = round(rating_float / 2)
                stars = "★" * num_stars + "☆" * (5 - num_stars)
        else: stars = "N/A"
    except (ValueError, TypeError): stars = "N/A"

    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            if poster_url and poster_url != "N/A":
                try:
                    img_response = requests.get(poster_url, timeout=10)
                    img_response.raise_for_status()
                    img = Image.open(BytesIO(img_response.content))
                    st.image(img, use_container_width=True)
                except Exception: st.image("https://placehold.jp/30/3d4070/ffffff/200x300.png?text=CineMatch-%20Movie%20Poster", use_container_width=True)
            else: st.image("https://placehold.jp/30/3d4070/ffffff/200x300.png?text=CineMatch-%20Movie%20Poster", use_container_width=True)

        with col2:
            st.markdown(f'<div class="movie-title">{display_title} ({display_year_str})</div>', unsafe_allow_html=True)
            try: votes_str = f"{int(float(str(display_votes).replace(',', ''))):,}" if display_votes != 'N/A' else 'N/A'
            except: votes_str = 'N/A'
            st.markdown(f'<span class="rating-stars">{stars}</span> <span class="movie-meta">{display_rating}/10 | {votes_str} votes</span>', unsafe_allow_html=True)
            meta_info = []
            if display_genre: meta_info.append(f"Genres: {display_genre}")
            if language: meta_info.append(f"Language: {language}")
            if country: meta_info.append(f"Country: {country}")
            if display_runtime and display_runtime != "N/A": meta_info.append(f"Runtime: {display_runtime}")
            for info in meta_info: st.markdown(f'<div class="movie-meta">{info}</div>', unsafe_allow_html=True)
            st.markdown('<div class="decoration-bar"></div>', unsafe_allow_html=True)
            st.markdown('<div class="movie-description">', unsafe_allow_html=True)
            st.markdown("Overview:"); st.markdown(f"{display_plot}"); st.markdown('</div>', unsafe_allow_html=True)

            summary_controls_area = st.container()
            summary_display_area = st.container()

            can_summarize = display_plot != "Plot summary not available." and GEMINI_API_KEY and imdb_id
            button_key = f"summarize_{imdb_id}" if imdb_id else f"summarize_{display_title}"

            summary_controls_area.button(
                "✨ Summarize Plot (AI)",
                key=button_key,
                disabled=not can_summarize,
                help="Click to generate an AI summary" if can_summarize else "Summary unavailable",
                on_click=handle_summarize_click,
                args=(imdb_id, display_plot, display_title)
            )

            if imdb_id and imdb_id in st.session_state.summaries:
                 summary_text = st.session_state.summaries[imdb_id]
                 if summary_text == "Generating...":
                      summary_display_area.info("Generating summary...")
                 else:
                      with summary_display_area.expander("Fun AI Summary", expanded=True):
                          st.markdown(f"{summary_text}")

            if not can_summarize and not GEMINI_API_KEY:
                 summary_controls_area.caption("AI Summary disabled (Gemini API key missing)")
            elif not can_summarize and display_plot == "Plot summary not available.":
                 summary_controls_area.caption("AI Summary unavailable (Plot missing)")

            if imdb_url: st.link_button("View on IMDb", imdb_url, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)


#This sets up the sidebar on the left.
#It has the title and the radio buttons for navigating between pages.
#It uses session state to remember which page the user is on.

st.sidebar.image("https://img.icons8.com/color/96/000000/clapperboard.png", width=80)
st.sidebar.title("CineMatch Pro")
st.session_state.current_page = st.sidebar.radio(
    "Navigation",
    ["Home", "Movie Recommendations", "AI Movie Finder", "Explore by Genre", "Decade Explorer", "About"],
    key='page_selector',
    index=["Home", "Movie Recommendations", "AI Movie Finder", "Explore by Genre", "Decade Explorer", "About"].index(st.session_state.current_page),
    label_visibility="collapsed"
)
page = st.session_state.current_page

#This is the main part of the app. 
#It shows different content based on the selected page.
#The Home page has a welcome message and a pie chart showing the distribution of genres in the dataset.
if page == "Home":
    st.markdown('<div style="text-align: center; padding: 2rem 0;"><h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 1rem;">CineMatch Pro</h1><p style="font-size: 1.5rem; color: #666; margin-bottom: 2rem;">Your AI-Powered Movie Recommendation System</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="movie-card" style="height: 100%;">', unsafe_allow_html=True)
        st.markdown("""<h2 style="color: #0A2647; margin-bottom: 1.5rem;">Discover Your Next Favorite Movie</h2> <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;"> CineMatch Pro uses advanced content analysis (TF-IDF on genres) and efficient Approximate Nearest Neighbors (ANN) search via Faiss to recommend movies based on similarity. It also leverages the Gemini AI for generating fun summaries and finding movies based on your mood! </p> <h3 style="color: #205295; margin: 1.5rem 0 1rem 0;">Key Features</h3> <ul style="font-size: 1.1rem; line-height: 1.6;"> <li><strong>Efficient Content Recommendations</strong>: Find similar movies quickly using Faiss ANN search.</li> <li><strong>AI Movie Finder (Gemini Powered)</strong>: Describe what you're in the mood for and let AI suggest movies.</li> <li><strong>AI Plot Summaries</strong>: Get fun, concise summaries generated by Gemini AI.</li> <li><strong>Genre Exploration</strong>: Discover top movies in your preferred genres.</li> <li><strong>Decade Explorer</strong>: Travel through cinema history by decade.</li> <li><strong>Real-time Movie Data</strong>: Integration with OMDb API for posters and details.</li> </ul> """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #205295; margin-bottom: 1rem;">Genre Distribution</h3>', unsafe_allow_html=True)
        try:
            if metadata is not None and 'genres' in metadata.columns:
                genre_counts_list = [g for genres in metadata['genres'].dropna() if isinstance(genres, str) for g in genres.split(',')]
                if genre_counts_list:
                    genre_counts = pd.Series(genre_counts_list).value_counts().head(10)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(genre_counts)))
                    ax.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', shadow=True, startangle=90, colors=colors)
                    ax.axis('equal'); st.pyplot(fig)
                else: st.write("No genre data.")
            else: st.write("'genres' column missing.")
        except Exception as e: st.error(f"Error generating genre plot: {e}")
        st.markdown('</div>', unsafe_allow_html=True)


#This is the Movie Recommendations page.
#User selects a movie, clicks a button, and gets similar movies using the get_recommendations_by_title function from utils_faiss.py.
 
elif page == "Movie Recommendations":
   
    st.markdown('<h1>Movie Recommendations</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Find movies similar to your favorites</p>', unsafe_allow_html=True)
    try: movie_titles = sorted(metadata['primaryTitle'].dropna().unique().tolist()) if metadata is not None else []
    except Exception as e: st.error(f"Error preparing movie list: {e}"); movie_titles = []

    with st.container():
        st.markdown('<div class="movie-card" style="padding: 1.5rem;">', unsafe_allow_html=True)
        selected_movie = st.selectbox("Search for a movie:", options=[""] + movie_titles, index=0, key="movie_search_input")
        search_button = st.button("Get Recommendations", key="search_button", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    if search_button and selected_movie:
        st.session_state.summaries = {}
        with st.spinner(f"Finding recommendations similar to '{selected_movie}'..."):
            # This is where the main recommendation happens using utils_faiss.py
            recommendations, matched_movie = utils.get_recommendations_by_title(selected_movie, model_data, n=10)
            st.session_state.recommendations = recommendations
            st.session_state.matched_movie = matched_movie
            if matched_movie is not None:
                st.session_state.omdb_info_selected = fetch_omdb_details_cached(api_key=OMDB_API_KEY, imdb_id=matched_movie['tconst'])
            else:
                st.session_state.omdb_info_selected = None
                st.error(f"Could not find '{selected_movie}'")
        st.rerun()

    if st.session_state.matched_movie is not None:
        year_str = str(int(st.session_state.matched_movie['startYear'])) if pd.notna(st.session_state.matched_movie['startYear']) else 'N/A'
        st.success(f"Showing recommendations based on: {st.session_state.matched_movie['primaryTitle']} ({year_str})")
        st.markdown('<h2 style="margin-top: 2rem;">Selected Movie</h2>', unsafe_allow_html=True)
        display_movie_card(st.session_state.matched_movie, st.session_state.omdb_info_selected)

        if st.session_state.recommendations is not None and not st.session_state.recommendations.empty:
            st.markdown('<h2 style="margin-top: 2rem;">Recommended Movies</h2>', unsafe_allow_html=True)
            for i, (_, rec_movie) in enumerate(st.session_state.recommendations.head(5).iterrows()):
                rec_omdb_info = fetch_omdb_details_cached(api_key=OMDB_API_KEY, imdb_id=rec_movie['tconst'])
                display_movie_card(rec_movie, rec_omdb_info)
                similarity_score = rec_movie.get("similarity", 0.0)
                st.markdown(f'<div style="text-align: right; margin-top: -1.5rem; margin-bottom: 1.5rem;"><span style="background-color: #205295; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem;">Similarity Score: {similarity_score:.3f}</span></div>', unsafe_allow_html=True)
        else:
             if selected_movie and search_button:
                  st.warning(f"Could not find sufficient recommendations similar to '{st.session_state.matched_movie['primaryTitle']}'.")

#This is the AI Movie Finder page.
#User types in what they want to watch, and Gemini suggests movies using the get_ai_recs_wrapper function.
#It also allows for optional filters like year, rating, and runtime.
elif page == "AI Movie Finder":

    st.markdown('<h1>AI Movie Finder (Powered by Gemini)</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Describe the kind of movie you want to watch</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="movie-card" style="padding: 1.5rem;">', unsafe_allow_html=True)
        user_prompt = st.text_area("Describe what you're in the mood for:", placeholder="Example: I want a heartwarming animated movie about friendship...", height=100, key="ai_prompt_input")
        with st.expander("Refine Search (Optional Filters)"):
            min_year = int(metadata['startYear'].min()) if metadata is not None else 1900
            max_year = int(metadata['startYear'].max()) if metadata is not None else 2025
            col1, col2 = st.columns(2)
            with col1:
                min_year_filter = st.number_input("Earliest Year:", min_value=min_year, max_value=max_year, value=max(min_year, 1980), key="ai_min_year_input")
                min_rating_filter = st.slider("Minimum Rating:", 1.0, 10.0, 6.0, 0.5, key="ai_min_rating_input")
            with col2:
                max_year_filter = st.number_input("Latest Year:", min_value=min_year, max_value=max_year, value=max_year, key="ai_max_year_input")
                max_runtime_filter = st.slider("Max Runtime (min):", 30, 240, 180, 15, key="ai_max_runtime_input")
        ai_button_disabled = not GEMINI_API_KEY
        if ai_button_disabled: st.warning("AI Movie Finder disabled: Gemini API key missing.")
        ai_button = st.button("Ask AI ✨", key="ai_search_button", type="primary", disabled=ai_button_disabled)
        st.markdown('</div>', unsafe_allow_html=True)

    if ai_button and user_prompt:
        st.session_state.summaries = {}
        filters_text = f"Year: {min_year_filter}-{max_year_filter}, Min Rating: {min_rating_filter:.1f}, Max Runtime: {max_runtime_filter} min"
        with st.spinner("Generating recommendations..."):
            # Calling the Gemini AI function
            st.session_state.gemini_recs = get_ai_recs_wrapper(GEMINI_API_KEY, user_prompt, filters_text)
            st.session_state.ai_filters = {
                'prompt': user_prompt,
                'filters_text': filters_text,
                'min_year': min_year_filter,
                'max_year': max_year_filter,
                'min_rating': min_rating_filter,
                'max_runtime': max_runtime_filter
            }
        st.rerun()

    if st.session_state.gemini_recs:
        if "Error:" in st.session_state.gemini_recs[0]:
            st.error(st.session_state.gemini_recs[0])
        elif not st.session_state.gemini_recs:
            st.warning("Gemini AI did not return any recommendations.")
        else:
            st.success("AI Suggestions:")
            if st.session_state.ai_filters:
                 st.markdown('<div class="movie-card" style="background-color: #f8f9fa; padding: 1rem; margin-top: 1rem;">', unsafe_allow_html=True)
                 st.markdown(f"Based on: '{st.session_state.ai_filters['prompt']}'")
                 st.markdown(f"Filters: {st.session_state.ai_filters['filters_text']}")
                 st.markdown("Suggestions:\n" + "\n".join([f"- {title}" for title in st.session_state.gemini_recs]))
                 st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<h2 style="margin-top: 2rem;">Displaying Found Movies</h2>', unsafe_allow_html=True)
            found_count = 0
            current_filters = st.session_state.ai_filters
            for title in st.session_state.gemini_recs:
                # Trying to find the movie suggested by AI in our database using utils_faiss.py
                matches = utils.get_movie_by_title(title, model_data)
                if not matches.empty:
                    movie = matches.iloc[0]
                    valid = True
                    if pd.notna(movie.get('startYear')): valid &= current_filters['min_year'] <= movie['startYear'] <= current_filters['max_year']
                    if pd.notna(movie.get('averageRating')): valid &= movie['averageRating'] >= current_filters['min_rating']
                    if pd.notna(movie.get('runtimeMinutes')): valid &= movie['runtimeMinutes'] <= current_filters['max_runtime']
                    if valid:
                        found_count += 1
                        omdb_info = fetch_omdb_details_cached(OMDB_API_KEY, imdb_id=movie['tconst'])
                        display_movie_card(movie, omdb_info)
                    else:
                         st.caption(f"{title} found, but excluded by filters.")
                else:
                    st.caption(f"Could not find {title} in local database.")
            if found_count == 0: st.warning("None of the AI suggestions matched filters or were found locally.")

#This is the Explore by Genre page.
#User selects a genre and year range, and gets top movies via the get_top_movies_by_genre function from utils_faiss.py.

elif page == "Explore by Genre":
    
    st.markdown('<h1>Explore Movies by Genre</h1>', unsafe_allow_html=True)
    all_genres = set()
    if metadata is not None and 'genres' in metadata.columns:
        for genres_str in metadata['genres'].dropna():
            if isinstance(genres_str, str): all_genres.update(g.strip() for g in genres_str.split(','))
    unique_genres = ['All'] + sorted(list(all_genres))

    with st.container():
        st.markdown('<div class="movie-card" style="padding: 1.5rem;">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1: selected_genre = st.selectbox("Select Genre:", unique_genres, key="genre_select_input")
        with col2:
            min_year = int(metadata['startYear'].min()) if metadata is not None else 1900
            max_year = int(metadata['startYear'].max()) if metadata is not None else 2025
            year_range = st.slider("Select Year Range:", min_value=min_year, max_value=max_year, value=(max(min_year, 1990), max_year), key="genre_year_input")
        find_button = st.button("Find Movies", key="genre_search_button", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    if find_button:
        st.session_state.summaries = {}
        with st.spinner(f"Finding top movies..."):
            # Getting top movies for the selected genre using utils_faiss.py
            top_movies = utils.get_top_movies_by_genre(selected_genre, year_range, model_data, n=20)
            st.session_state.genre_results = {
                'genre': selected_genre,
                'years': year_range,
                'movies': top_movies
            }
        st.rerun()

    if st.session_state.genre_results:
        results = st.session_state.genre_results
        top_movies = results['movies']
        if top_movies is not None and not top_movies.empty:
            st.markdown(f'<h2 style="margin-top: 2rem;">Top {results["genre"]} Movies ({results["years"][0]}-{results["years"][1]})</h2>', unsafe_allow_html=True)
            with st.container():
                 st.markdown('<div class="movie-card" style="text-align: center; padding: 1.5rem; background-color: #f8f9fa; margin-bottom: 2rem;">', unsafe_allow_html=True)
                 st.markdown('<h3 style="color: #205295;">Movie Roulette</h3>', unsafe_allow_html=True); st.markdown('<p>Can\'t decide? Spin the wheel!</p>', unsafe_allow_html=True)
                 if st.button("Spin the Wheel!", key="roulette"):
                     if not top_movies.empty:
                         random_movie = top_movies.sample(1).iloc[0]; st.write(f"Roulette pick: {random_movie['primaryTitle']}")
                         omdb_info_random = fetch_omdb_details_cached(api_key=OMDB_API_KEY, imdb_id=random_movie['tconst'])
                         display_movie_card(random_movie, omdb_info_random)
                     else: st.warning("Not enough movies.")
                 st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<h3 style="margin-top: 2rem;">Featured Movies</h3>', unsafe_allow_html=True)
            for i, (_, movie) in enumerate(top_movies.head(5).iterrows()):
                omdb_info = fetch_omdb_details_cached(api_key=OMDB_API_KEY, imdb_id=movie['tconst'])
                display_movie_card(movie, omdb_info)
            if len(top_movies) > 5:
                with st.expander(f"View More ({len(top_movies) - 5} movies)"):
                    for i, (_, movie) in enumerate(top_movies[5:].iterrows()):
                        omdb_info = fetch_omdb_details_cached(api_key=OMDB_API_KEY, imdb_id=movie['tconst'])
                        display_movie_card(movie, omdb_info)
        else:
             if find_button:
                  st.warning(f"No '{results['genre']}' movies found for {results['years'][0]}-{results['years'][1]}.")

#This is the Decade Explorer page.
#User selects a decade and sees some stats and top movies from that time via the get_top_movies_by_decade function from utils_faiss.py.
elif page == "Decade Explorer":
    st.markdown('<h1>Decade Explorer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Discover cinema through the decades</p>', unsafe_allow_html=True)
    if metadata is not None and 'decade_str' in metadata.columns:
        decades = sorted(metadata['decade_str'].dropna().unique())
    else: st.error("Decade data unavailable."); st.stop(); decades = []
    if not decades: st.warning("No decade information.")
    else:
        with st.container():
            st.markdown('<div class="movie-card" style="padding: 1.5rem;">', unsafe_allow_html=True)
            selected_decade = st.selectbox("Select a Decade:", decades, key="decade_select_input")
            explore_button = st.button("Explore Decade", key="decade_search_button", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)

        if explore_button:
            st.session_state.summaries = {}
            with st.spinner(f"Loading movies from the {selected_decade}..."):
                # Just filtering the loaded metadata here
                decade_movies = metadata[metadata['decade_str'] == selected_decade].copy()
                st.session_state.decade_results = {
                    'decade': selected_decade,
                    'movies': decade_movies
                }
            st.rerun()

        if st.session_state.decade_results:
            results = st.session_state.decade_results
            decade_movies = results['movies']
            if not decade_movies.empty:
                st.markdown(f'<h2 style="margin-top: 2rem;">Cinema Snapshot: The {results["decade"]}</h2>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric(label="Movies in Dataset", value=f"{len(decade_movies):,}")
                with col2: avg_rating = decade_movies['averageRating'].mean() if 'averageRating' in decade_movies and decade_movies['averageRating'].notna().any() else np.nan; st.metric(label="Avg. Rating", value=f"{avg_rating:.2f}/10" if pd.notna(avg_rating) else "N/A")
                with col3: avg_runtime = decade_movies['runtimeMinutes'].mean() if 'runtimeMinutes' in decade_movies and decade_movies['runtimeMinutes'].notna().any() else np.nan; st.metric(label="Avg. Runtime", value=f"{avg_runtime:.0f} min" if pd.notna(avg_runtime) else "N/A")
                st.markdown(f'<h2 style="margin-top: 2rem;">Highly Rated Movies of the {results["decade"]}</h2>', unsafe_allow_html=True)
                sort_cols, sort_asc = [], []
                if 'averageRating' in decade_movies.columns: sort_cols.append('averageRating'); sort_asc.append(False)
                if 'popularity_score' in decade_movies.columns: sort_cols.append('popularity_score'); sort_asc.append(False)
                if not sort_cols: sort_cols = ['primaryTitle']; sort_asc = [True]
                top_decade_movies = decade_movies.sort_values(sort_cols, ascending=sort_asc, na_position='last').head(10)
                for i, (_, movie) in enumerate(top_decade_movies.iterrows()):
                    omdb_info = fetch_omdb_details_cached(api_key=OMDB_API_KEY, imdb_id=movie['tconst'])
                    st.markdown(f'<div style="position: relative;">', unsafe_allow_html=True)
                    st.markdown(f'<div style="position: absolute; top: 1rem; left: -0.8rem; background-color: #0A2647; color: white; width: 2.5rem; height: 2.5rem; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; z-index: 10; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{i+1}</div>', unsafe_allow_html=True)
                    display_movie_card(movie, omdb_info)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                 if explore_button:
                      st.warning(f"No movies found for the {results['decade']}.")
#This is the About page.
#It just shows text explaining how the app works and what is it using.
elif page == "About":
  
    st.markdown('<h1>About CineMatch Pro</h1>', unsafe_allow_html=True)
    st.markdown('<div class="movie-card" style="padding: 2rem;">', unsafe_allow_html=True)
    st.markdown(""" <h2 style="color: #0A2647; margin-bottom: 1.5rem;">How It Works</h2> <p style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem;"> CineMatch Pro leverages efficient content-based filtering and AI-driven features: </p> <ul style="font-size: 1.1rem; line-height: 1.6; margin-bottom: 2rem;"> <li><strong>Content Feature Extraction</strong>: Analyzes movie features like genres (using TF-IDF weighting), year, runtime, ratings, and crew counts. Numerical features are scaled.</li> <li><strong>Efficient Similarity Search (Faiss)</strong>: Builds an Approximate Nearest Neighbors (ANN) index using Faiss on the combined features for fast retrieval of similar movies.</li> <li><strong>Generative AI (Google Gemini)</strong>: Uses the Gemini 1.5 Flash model to generate fun plot summaries and provide movie recommendations based on natural language prompts.</li> <li><strong>API Integration</strong>: Connects with the OMDb API for fetching up-to-date movie details and posters.</li> </ul> <div style="height: 2px; background: linear-gradient(90deg, #0A2647, #205295, #2C74B3); margin: 2rem 0;"></div> <h2 style="color: #0A2647; margin-bottom: 1.5rem;">Features</h2> <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;"> <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"> <h3 style="color: #205295; margin-bottom: 1rem;">Movie Recommendations</h3> <p>Quickly find movies similar to your favorites using efficient Faiss-powered content similarity.</p> </div> <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"> <h3 style="color: #205295; margin-bottom: 1rem;">AI Movie Finder</h3> <p>Describe what you're in the mood for and get personalized suggestions powered by Google Gemini.</p> </div> <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"> <h3 style="color: #205295; margin-bottom: 1rem;">AI Summaries</h3> <p>Get fun, concise plot summaries generated on-demand by Google Gemini.</p> </div> <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"> <h3 style="color: #205295; margin-bottom: 1rem;">Genre Explorer</h3> <p>Browse top movies by genre and year to discover new favorites in categories you love.</p> </div> <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);"> <h3 style="color: #205295; margin-bottom: 1rem;">Decade Explorer</h3> <p>Discover classic films from different eras and see how cinema has evolved over time.</p> </div> </div> <div style="height: 2px; background: linear-gradient(90deg, #0A2647, #205295, #2C74B3); margin: 2rem 0;"></div> <h2 style="color: #0A2647; margin-bottom: 1.5rem;">Technologies Used</h2> <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem;"> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Python</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Pandas & NumPy</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Scikit-learn</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Faiss</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Google Gemini API</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Streamlit</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">OMDb API</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Matplotlib</span> <span style="background-color: #205295; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-size: 1rem;">Pillow</span> </div> """, unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-top: 3rem; color: #666;">', unsafe_allow_html=True)

#This is the footer of the app. 
#It shows the developer credits and course info.
st.sidebar.markdown('<div style="height: 2px; background: linear-gradient(90deg, #205295, #2C74B3, #205295); margin: 2rem 0 1rem 0;"></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 0 1rem;">', unsafe_allow_html=True)
st.sidebar.markdown('''
<b>Developed by</b><br>
<a href="https://www.linkedin.com/in/pateldhruvin" target="_blank" style="color: #ADD8E6;">Dhruvin Patel</a> |
<a href="https://www.linkedin.com/in/willxu425" target="_blank" style="color: #ADD8E6;">Hao Xu</a> |
<a href="https://www.linkedin.com/in/maleeha-babar-8a830b50" target="_blank" style="color: #ADD8E6;">Maleeha Babar</a>
''', unsafe_allow_html=True)
st.sidebar.markdown('<p style="margin-top: 0.5rem; font-size: 0.8rem;">Part of Machine Learning Course at Tulane University</p>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 0 1rem;">', unsafe_allow_html=True)