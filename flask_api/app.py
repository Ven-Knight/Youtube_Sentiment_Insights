import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import io
import os
import re
import string
import logging
import html
import contractions
import nltk
import emoji
import unidecode
import pkg_resources
import pickle
import mlflow
import json
import matplotlib.pyplot   as plt
import matplotlib.dates    as mdates
import numpy               as np
import pandas              as pd
from bs4                   import BeautifulSoup
from symspellpy.symspellpy import SymSpell, Verbosity
from wordcloud             import WordCloud
from nltk.corpus           import stopwords
from nltk.tokenize         import word_tokenize
from nltk.stem             import WordNetLemmatizer, PorterStemmer
from mlflow.tracking       import MlflowClient
from flask_cors            import CORS
from flask                 import Flask, request, jsonify, send_file


import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download required NLTK data for NLP
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt_tab')

# Initialize SymSpell for spelling correction
sym_spell  = SymSpell(max_dictionary_edit_distance=2)
dict_path  = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'} # retaining important ones for sentiment analysis

print("NLTK resources downloaded and NLP tools initialized")


# Define clean_comment function
# def clean_comment(text):
#     """Apply cleaning transformations to a comment."""
#     corrected_text = ""                    # Initialize to avoid reference error

#     try:
#         """Step-by-step text cleaning for NLP tasks."""

#         # Step 1️⃣: Convert Tensor to string if needed
#         # if isinstance(text, tf.Tensor):
#         #     text = text.numpy().decode("utf-8")

#         # Step 2️⃣: Remove HTML tags
#         text = BeautifulSoup(text, "html.parser").get_text()

#         # Step 3️⃣: Expand contractions (e.g., "don't" → "do not")
#         text = contractions.fix(text)
    
#         # Step 4️⃣: Replace hyphens with spaces
#         text = re.sub(r"-", " ", text)

#         # Step 5️⃣: Remove special characters (except basic punctuation)
#         text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)

#         # Step 6️⃣: Remove newline characters
#         text = text.replace('\n', ' ')

#         # Step 7️⃣: Remove URLs
#         url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
#         text = re.sub(url_pattern, '', text)

#         # Step 8️⃣: Convert to lowercase
#         text = text.lower()

#         # Step 9️⃣: Correct misspellings using SymSpell
#         words          = text.split()
#         corrected_text = " ".join([
#                                     sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)[0].term
#                                     if   sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
#                                     else word for word in words
#                                 ])

#         # Step 🔟: Remove extra whitespaces
#         corrected_text = re.sub(r"\s+", " ", corrected_text).strip()

#         # Step 1️⃣1️⃣: Normalize Unicode characters (e.g., accented letters → plain text)
#         corrected_text = unidecode.unidecode(corrected_text)

#         # Step 1️⃣2️⃣: Remove emojis and non-ASCII characters
#         corrected_text = emoji.replace_emoji(corrected_text, replace="")

#         return corrected_text

#     except Exception as e:
#         print(f"Error in cleaning comment: {e}")
#         return corrected_text

def clean_comment(text):
    corrected_text = ""
    try:
        logger.info(f"🔹 Raw input: {text}")

        text = BeautifulSoup(text, "html.parser").get_text()
        text = contractions.fix(text)
        text = re.sub(r"[–—]", " ", text)  # handle em-dash
        text = re.sub(r"-", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
        text = text.replace('\n', ' ')
        text = re.sub(r"http[s]?://\S+", '', text)
        text = text.lower()

        words = text.split()
        corrected_text = " ".join([
            sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)[0].term
            if sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            else word for word in words
        ])

        corrected_text = re.sub(r"\s+", " ", corrected_text).strip()
        corrected_text = unidecode.unidecode(corrected_text)
        corrected_text = emoji.replace_emoji(corrected_text, replace="")

        logger.info(f"✅ Cleaned text: {corrected_text}")
        return corrected_text

    except Exception as e:
        logger.error(f"❌ Error in clean_comment: {e}")
        logging.exception(e)
        return corrected_text
    

# Define the pre_processing function
# def preprocess_comment(cleaned_text):
#     """Apply pre_processing transformations to a comment."""
#     processed_text = ""                       # Initialize to avoid reference error

#     try:
#         """Step-by-step NLP preprocessing: tokenization, stopword removal, lemmatization."""

#         # Step 1️⃣: Tokenize the cleaned text
#         tokens = word_tokenize(cleaned_text)

#         # Step 2️⃣: Remove stopwords
#         tokens = [word for word in tokens if word not in stop_words]

#         # Step 3️⃣: Lemmatize tokens
#         lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

#         # Step 4️⃣: (Optional) Stemming — currently commented out
#         # stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

#         # Step 5️⃣: Reconstruct the processed sentence
#         processed_text = " ".join(lemmatized_tokens)

#         return processed_text

#     except Exception as e:
#         print(f"Error in preprocessing comment: {e}")
#         return processed_text

def preprocess_comment(cleaned_text):
    processed_text = ""
    try:
        logger.info(f"🔹 Preprocessing input: {cleaned_text}")

        tokens = word_tokenize(cleaned_text)
        logger.info(f"🔹 Tokens: {tokens}")

        tokens = [word for word in tokens if word not in stop_words]
        logger.info(f"🔹 After stopword removal: {tokens}")

        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        logger.info(f"🔹 Lemmatized tokens: {lemmatized_tokens}")

        processed_text = " ".join(lemmatized_tokens)

        if not processed_text.strip():
            logger.warning("⚠️ Preprocessed text is empty, falling back to cleaned text")
            processed_text = cleaned_text

        logger.info(f"✅ Final preprocessed text: {processed_text}")
        return processed_text

    except Exception as e:
        logger.error(f"❌ Error in preprocess_comment: {e}")
        logging.exception(e)
        return cleaned_text


# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    try:
        print(f"Loading model: {model_name} version: {model_version}")
        mlflow.set_tracking_uri("http://ec2-13-233-244-190.ap-south-1.compute.amazonaws.com:5000/")   # MLflow tracking URI
        client = MlflowClient()

        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        print("Model loaded successfully")
        print(f"Model type: {type(model)}")

        # Validate vectorizer file path
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")

        print(f"Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        print("Vectorizer loaded successfully")

        return model, vectorizer

    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")
        return None, None




@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict', methods=['POST'])
def predict():
    data     = request.json
    comments = data.get('comments')
    # print("i am the comment: ",comments)
    # print("i am the comment type: ",type(comments))
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        cleaned_comments      = [clean_comment     (comment) for comment in comments]
        preprocessed_comments = [preprocess_comment(comment) for comment in cleaned_comments]
        
        # Transform comments using the vectorizer
        transformed_comments  = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments        = transformed_comments.toarray()          # Convert to dense array
        feature_names         = vectorizer.get_feature_names_out()
        dense_df              = pd.DataFrame(dense_comments, columns=feature_names) # to be in sync with signature
        
        # Make predictions
        predictions           = model.predict(dense_df).tolist()       # Convert to list
        
        # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data          = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments              = [item['text']      for item in comments_data]
        timestamps            = [item['timestamp'] for item in comments_data]

        # Preprocess each comment before vectorizing
        cleaned_comments      = [clean_comment     (comment) for comment in comments]
        preprocessed_comments = [preprocess_comment(comment) for comment in cleaned_comments]
        
        # Transform comments using the vectorizer
        transformed_comments  = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments        = transformed_comments.toarray()          # Convert to dense array
        feature_names         = vectorizer.get_feature_names_out()
        dense_df              = pd.DataFrame(dense_comments, columns=feature_names) # to be in sync with signature
        
        # Make predictions
        predictions           = model.predict(dense_df).tolist()        # Convert to list

        
        # Convert predictions to strings for consistency
        predictions           = [str(pred) for pred in predictions]

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data             = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels           = ['Positive', 'Neutral', 'Negative']
        sizes            = [
                               int(sentiment_counts.get('1',  0)),
                               int(sentiment_counts.get('0',  0)),
                               int(sentiment_counts.get('-1', 0))
                           ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
                   sizes,
                   labels     = labels,
                   colors     = colors,
                   autopct    = '%1.1f%%',
                   startangle = 140,
                   textprops  = {'color': 'w'}
               )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data     = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        cleaned_comments      = [clean_comment     (comment) for comment in comments]
        preprocessed_comments = [preprocess_comment(comment) for comment in cleaned_comments]
        
        # Combine all comments into a single string
        text                  = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud             = WordCloud(
                                            width            = 800,
                                            height           = 400,
                                            background_color = 'black',
                                            colormap         = 'Blues',
                                            stopwords        = set(stopwords.words('english')),
                                            collocations     = False
                                         ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io                = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data             = request.get_json()
        sentiment_data   = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df               = pd.DataFrame(sentiment_data)
        df['timestamp']  = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment']  = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts   = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals   = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
                     -1: 'red',      # Negative sentiment
                      0: 'gray',     # Neutral  sentiment
                      1: 'green'     # Positive sentiment
                 }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                       monthly_percentages.index,
                       monthly_percentages[sentiment_value],
                       marker    = 'o',
                       linestyle = '-',
                       label     = sentiment_labels[sentiment_value],
                       color     = colors[sentiment_value]
                    )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator  (mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500
    
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/debug', methods=['POST'])
def debug():
    data         = request.json
    text         = data.get('text', '')
    cleaned      = clean_comment(text)
    preprocessed = preprocess_comment(cleaned)
    vec          = vectorizer.transform([preprocessed])
    df           = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names_out())
    pred         = model.predict(df).tolist()
    return jsonify({
                      "comment"          : text,
                      "cleaned"          : cleaned,
                      "preprocessed"     : preprocessed,
                      "vectorized_shape" : vec.shape,
                      "nonzero_features" : int((vec != 0).sum()),   # tells how many features are actually active
                      "prediction"       : pred
                   })

if __name__ == '__main__':
    try:
        # Load model configuration from JSON file
        with open("model_config.json", "r") as f:
            config        = json.load(f)

        model_name        = config["model_name"]
        model_version     = config["model_version"]
        vectorizer_path   = config["vectorizer_path"]

        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer(model_name, model_version, vectorizer_path)
        
        print(f"Model '{model_name}' version {model_version} loaded successfully")

    except Exception as e:
        print(f"Model loading failed: {e}")
        model, vectorizer = None, None

    print("Starting Flask app on port 8080...")
    app.run(host='0.0.0.0', port=8080)

