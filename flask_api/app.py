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

# ────────────────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ────────────────────────────────────────────────────────────────────────────────────────

logger          = logging.getLogger('flask_api')
logger            .setLevel(logging.DEBUG)

# Console handler for DEBUG and above
console_handler = logging.StreamHandler()
console_handler   .setLevel(logging.DEBUG)

# File handler for ERROR and above
file_handler    = logging.FileHandler('flask_api_errors.log')
file_handler      .setLevel(logging.ERROR)

# Formatter
formatter       = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler   .setFormatter(formatter)
file_handler      .setFormatter(formatter)

# Attach handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ────────────────────────────────────────────────────────────────────────────────────────
# Initialize Flask app
# ────────────────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)            # Enable CORS for all routes
logger.info(" ✅ Flask app initialized & CORS enabled for all routes")


# ────────────────────────────────────────────────────────────────────────────────────────
# Download required NLTK data
# ────────────────────────────────────────────────────────────────────────────────────────
try:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("punkt_tab")
    logger.info(" ✅ NLTK resources downloaded successfully")
except Exception as e:
    logger.error("❌ Failed to download NLTK resources")
    logger.exception(e)

# Initialize SymSpell
try:
    sym_spell  = SymSpell(max_dictionary_edit_distance=2)
    dict_path  = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
    logger.info(f" ✅ SymSpell initialized with {len(sym_spell._words)} words")
except Exception as e:
    logger.error("❌ Failed to initialize SymSpell")
    logger.exception(e)

# Initialize NLP tools
try:
    lemmatizer = WordNetLemmatizer()
    stemmer    = PorterStemmer()
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    logger.info(f" ✅ NLP tools initialized. Stopwords count: {len(stop_words)}")
except Exception as e:
    logger.error("❌ Failed to initialize NLP tools")
    logger.exception(e)



# ────────────────────────────────────────────────────────────────────────────────────────
# Define clean_comment function
# ────────────────────────────────────────────────────────────────────────────────────────

def clean_comment(text):
    """Apply cleaning transformations to a comment."""
    corrected_text = ""                    # Initialize to avoid reference error

    try:
        """Step-by-step text cleaning for NLP tasks."""
        logger.info(f" 🔹 Raw comment          : {text}")

        # Step 1️⃣: Convert Tensor to string if needed
        # if isinstance(text, tf.Tensor):
        #     text = text.numpy().decode("utf-8")

        # Step 2️⃣: Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Step 3️⃣: Expand contractions (e.g., "don't" → "do not")
        text = contractions.fix(text)
    
        # Step 4️⃣: Replace hyphens with spaces
        text = re.sub(r"-", " ", text)

        # Step 5️⃣: Remove special characters (except basic punctuation)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)

        # Step 6️⃣: Remove newline characters
        text = text.replace('\n', ' ')

        # Step 7️⃣: Remove URLs
        url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        text = re.sub(url_pattern, '', text)

        # Step 8️⃣: Convert to lowercase
        text = text.lower()

        # Step 9️⃣: Correct misspellings using SymSpell
        words          = text.split()
        corrected_text = " ".join([
                                    sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)[0].term
                                    if   sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
                                    else word for word in words
                                ])

        # Step 🔟: Remove extra whitespaces
        corrected_text = re.sub(r"\s+", " ", corrected_text).strip()

        # Step 1️⃣1️⃣: Normalize Unicode characters (e.g., accented letters → plain text)
        corrected_text = unidecode.unidecode(corrected_text)

        # Step 1️⃣2️⃣: Remove emojis and non-ASCII characters
        corrected_text = emoji.replace_emoji(corrected_text, replace="")

        logger.info(f" 🔹 Cleaned comment      : {corrected_text}")
        return corrected_text

    except Exception as e:
        logger.error("❌ Error in clean_comment")
        logger.exception(e)
        return corrected_text

# ──────────────────────────────────────────────────────────────────────────────────────
# Define preprocess_comment function
# ──────────────────────────────────────────────────────────────────────────────────────

def preprocess_comment(cleaned_text):
    """Apply NLP preprocessing: tokenization, stopword removal, lemmatization."""
    processed_text = ""                       # Initialize to avoid reference error

    try:
        
        # Step 1️⃣: Tokenize the cleaned text
        tokens = word_tokenize(cleaned_text)

        # Step 2️⃣: Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]

        # Step 3️⃣: Lemmatize tokens
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Step 4️⃣: (Optional) Stemming — currently commented out
        # stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]

        # Step 5️⃣: Reconstruct the processed sentence
        processed_text = " ".join(lemmatized_tokens)

        # Step 6️⃣: Fallback if empty
        if not processed_text.strip():
            logger.warning("⚠️ Preprocessed text is empty, falling back to cleaned text")
            processed_text = cleaned_text

        logger.info(f" ✅ preprocessed comment : {processed_text}")
        return processed_text

    except Exception as e:
        logger.error("❌ Error in preprocess_comment")
        logger.exception(e)
        return cleaned_text  # fallback to cleaned version


# ──────────────────────────────────────────────────────────────────────────────────────
# Load the model and vectorizer from the model registry and local storage
# ──────────────────────────────────────────────────────────────────────────────────────
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://ec2-13-233-244-190.ap-south-1.compute.amazonaws.com:5000/")   # MLflow tracking URI
        client = MlflowClient()

        # Load model from MLflow registry
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        logger.debug(f"📦 Model type: {type(model)}")
        logger.info(f" ✅ Model '{model_name}' version {model_version} loaded successfully")
            
        # Validate vectorizer file path
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")

        logger.info(f" 🔄 Loading vectorizer from: {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info(" ✅ Vectorizer loaded successfully")

        return model, vectorizer

    except Exception as e:
        logger.error("❌ Error loading model or vectorizer")
        logger.exception(e)
        return None, None


# ──────────────────────────────────────────────────────────────────────────────────────
# Define root route
# ──────────────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    logger.info(" 📥 Root endpoint '/' accessed")
    return "✅ Welcome to the FLASK Sentiment API"

# ──────────────────────────────────────────────────────────────────────────────────────
# Define prediction route
# ──────────────────────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    data     = request.json
    comments = data.get('comments')

    logger.info(" 📥 /predict endpoint accessed")
        
    if not comments:
        logger.warning("⚠️ No comments provided in request")
        return jsonify({"error": "No comments provided"}), 400


    try:
        preprocessed_comments = []

        for idx, comment in enumerate(comments):
            logger.info(f" 🧠 Processing comment #{idx+1}")    
            # Clean and preprocess        
            cleaned           = clean_comment(comment)            
            preprocessed      = preprocess_comment(cleaned)     
            preprocessed_comments.append(preprocessed)

        # Transform comments using the vectorizer
        transformed_comments  = vectorizer.transform(preprocessed_comments)

        # Convert the sparse matrix to dense format
        dense_comments        = transformed_comments.toarray()          # Convert to dense array
        feature_names         = vectorizer.get_feature_names_out()
        dense_df              = pd.DataFrame(dense_comments, columns=feature_names) # to be in sync with signature
        
        # Make predictions
        predictions           = model.predict(dense_df).tolist()       # Convert to list
        logger.info(" ✅ Predictions successfully generated by model")

        # Convert predictions to strings for consistency
        # predictions = [str(pred) for pred in predictions]

    except Exception as e:
        logger.error("❌ Prediction pipeline failed")
        logger.exception(e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    
    # Return the response with original comments and predicted sentiments
    response = [
                    {"comment": comment, "sentiment": sentiment} 
                    for comment, sentiment in zip(comments, predictions)
               ]
    return jsonify(response)


# ──────────────────────────────────────────────────────────────────────────────────────
# Define timestamped prediction route
# ──────────────────────────────────────────────────────────────────────────────────────

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data          = request.json
    comments_data = data.get('comments')
    
    logger.info(" 📥 /predict_with_timestamps endpoint accessed")
    logger.debug(f"🔹 Received payload: {comments_data}")

    if not comments_data:
        logger.warning("⚠️ No comments provided in request")
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

        logger.info("✅ Predictions successfully generated by model")
    
    except Exception as e:
        logger.error("❌ Prediction pipeline failed")
        logger.exception(e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    
    # construct and return the response with original comments, predicted sentiments, and timestamps
    response = [
                    {"comment": comment, "sentiment": sentiment, "timestamp": timestamp} 
                    for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
               ]
    return jsonify(response)



# ──────────────────────────────────────────────────────────────────────────────────────
# Define chart generation route
# ──────────────────────────────────────────────────────────────────────────────────────


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        logger.info(" 📥 /generate_chart endpoint accessed")

        data             = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            logger.warning("⚠️ No sentiment counts provided in request")
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
        logger.info(" ✅ Chart generated successfully")

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        logger.error("❌ Error in /generate_chart")
        logger.exception(e)
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500



# ──────────────────────────────────────────────────────────────────────────────────────
# Define word cloud generation route
# ──────────────────────────────────────────────────────────────────────────────────────

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        logger.info(" 📥 /generate_wordcloud endpoint accessed")

        data     = request.get_json()
        comments = data.get('comments')

        if not comments:
            logger.warning("⚠️ No comments provided in request")
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
        
        logger.info(" ✅ Word cloud generated successfully")

        # Save the word cloud to a BytesIO object
        img_io                = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        logger.error("❌ Error in /generate_wordcloud")
        logger.exception(e)
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

# ──────────────────────────────────────────────────────────────────────────────────────
# Define sentiment trend graph generation route
# ──────────────────────────────────────────────────────────────────────────────────────
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        logger.info(" 📥 /generate_trend_graph endpoint accessed")

        data             = request.get_json()
        sentiment_data   = data.get('sentiment_data')

        if not sentiment_data:
            logger.warning("⚠️ No sentiment data provided in request")
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
        logger.info(" ✅ Trend graph generated successfully")

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        logger.error("❌ Error in /generate_trend_graph")
        logger.exception(e)
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

# ──────────────────────────────────────────────────────────────────────────────────────
# Define health_check route
# ──────────────────────────────────────────────────────────────────────────────────────
@app.route('/health')
def health():
    logger.info(" 📥 /health endpoint accessed")
    return jsonify({"status": "ok", "message": "API is healthy and running"})


# ──────────────────────────────────────────────────────────────────────────────────────
# Define debug route for introspection
# ──────────────────────────────────────────────────────────────────────────────────────

@app.route('/debug', methods=['POST'])
def debug():
    try:
        logger.info(" 📥 /debug endpoint accessed")

        data             = request.json
        comments         = data.get('comments')
        
        if not comments:
            logger.warning("⚠️ No comments provided in request")
            return jsonify({"error": "No comments provided"}), 400

        debug_responses      = []

        for index, text in enumerate(comments):
            logger.info(f" 🧠 Processing comment #{index+1}")

            # Clean and preprocess
            cleaned          = clean_comment     (text)
            preprocessed     = preprocess_comment(cleaned)

            # Vectorize
            vec              = vectorizer.transform([preprocessed])
            vectorized_shape = vec.shape
            nonzero_features = int((vec != 0).sum())
            logger.debug(f"🔹 Vectorized shape     : {vectorized_shape}")
            logger.debug(f"🔹 Non-zero features    : {nonzero_features}")

            # Convert to DataFrame
            df               = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names_out())

            # Predict
            pred             = model.predict(df).tolist()
            logger.info(" ✅ Prediction generated successfully")

            # Construct response
            debug_responses.append({
                                     "comment"          : text,
                                     "cleaned"          : cleaned,
                                     "preprocessed"     : preprocessed,
                                     "vectorized_shape" : vectorized_shape,
                                     "nonzero_features" : nonzero_features,
                                     "prediction"       : pred
                                  })
            
        return jsonify(debug_responses)

    except Exception as e:
        logger.error("❌ Error in /debug route")
        logger.exception(e)
        return jsonify({"error": f"Debug failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────────────────────────────
# Entry point: Load model and start Flask app
# ──────────────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        logger.info(" 🚀 Starting model and vectorizer initialization")

        # Load model configuration from JSON file
        with open("model_config.json", "r") as f:
            config        = json.load(f)

        model_name        = config["model_name"]
        model_version     = config["model_version"]
        vectorizer_path   = config["vectorizer_path"]
        logger.debug(f"🔹 Loaded model_config.json : {config}")

        # Load model and vectorizer
        model, vectorizer = load_model_and_vectorizer(model_name, model_version, vectorizer_path)
        
        if model      == None:
            logger.warning("⚠️ Model is None after loading")
        if vectorizer == None:
            logger.warning("⚠️ vectorizer is None after loading")


    except Exception as e:
        logger.error("❌ Model loading failed")
        logger.exception(e)
        model, vectorizer = None, None


    logger.info(" 🌐 Starting Flask app on port 8080")
    app.run(host='0.0.0.0', port=8080)

