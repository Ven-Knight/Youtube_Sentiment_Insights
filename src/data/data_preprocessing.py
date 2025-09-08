
import numpy               as np
import pandas              as pd
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

from bs4                   import BeautifulSoup
from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.tokenize         import word_tokenize
from nltk.corpus           import stopwords
from nltk.stem             import WordNetLemmatizer, PorterStemmer
from collections           import Counter

# logging configuration
logger          = logging.getLogger('data_preprocessing')
logger            .setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler   .setLevel('DEBUG')

file_handler    = logging.FileHandler('preprocessing_errors.log')
file_handler      .setLevel('ERROR')

formatter       = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler   .setFormatter(formatter)
file_handler      .setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Download required NLTK data for NLP
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize SymSpell for spelling correction
sym_spell  = SymSpell(max_dictionary_edit_distance=2)
dict_path  = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'} # retaining important ones for sentiment analysis



# Define clean_comment function
def clean_comment(text):
    """Apply cleaning transformations to a comment."""
    try:
        """Step-by-step text cleaning for NLP tasks."""

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

        return corrected_text

    except Exception as e:
        logger.error(f"Error in cleaning comment: {e}")
        return corrected_text

# Define the pre_processing function
def preprocess_comment(cleaned_text):
    """Apply pre_processing transformations to a comment."""
    try:
        """Step-by-step NLP preprocessing: tokenization, stopword removal, lemmatization."""

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

        return processed_text

    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return processed_text


def normalize_text(df):
    """Apply preprocessing to the text data in the dataframe."""
    try:
        # 1️⃣ Word count in raw comment
        df['word_count_raw']      = df['raw_comment']      .apply(lambda x: len(str(x).split()))

        # 2️⃣ Character count in raw comment (includes spaces and punctuation)
        df['char_count_raw']      = df['raw_comment']      .apply(lambda x: len(str(x)))

        # 3️⃣ Unique word count in raw comment
        df['num_unique_words']    = df['raw_comment']      .apply(lambda x: len(set(str(x).split())))

        # 4️⃣ Count of fully uppercase words in raw comment
        df['num_upper_words']     = df['raw_comment']      .apply(lambda x: len([word for word in str(x).split() if word.isupper()]))

        # 5️⃣ Punctuation count in raw comment
        df['num_punctuation_raw'] = df['raw_comment']      .apply(lambda x: sum(1 for char in str(x) if char in string.punctuation))

        # 6️⃣ Clean the raw comment (e.g., remove unwanted characters)
        df['cleaned_comment']     = df['raw_comment']      .apply(clean_comment)

        # 7️⃣ Stop word count in cleaned comment
        df['num_stop_words']      = df['cleaned_comment']  .apply(lambda x: len([word for word in x.split() if word in stop_words]))

        # 8️⃣ Apply NLP preprocessing (e.g., lemmatization, lowercasing)
        df['processed_comment']   = df['cleaned_comment']  .apply(preprocess_comment)

        # 9️⃣ Final word count after preprocessing
        df['final_word_count']    = df['processed_comment'].apply(lambda x: len(x.split()))

        # 🔟 Final character count in processed comment (excluding spaces)
        df['final_char_count']    = df['processed_comment'].apply(lambda x: len(x.replace(" ", "")))

        # 11. After processing some columns may end up with empty string (not Nan), below will clean them
        df.replace('', np.nan, inplace=True)                # Convert empty strings to NaN
        df.dropna(inplace=True)                             # drops rows with missing values
        
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Ensure the directory is created
        logger.debug(f"Directory {interim_data_path} created or already exists")

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv (os.path.join(interim_data_path, "test_processed.csv"),  index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

def main():
    try:
        logger.debug("Starting data pre_processing...")
        
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data  = pd.read_csv('./data/raw/test.csv')
        logger.debug('Raw data fetched successfully')

        # Preprocess the data
        train_processed_data = normalize_text(train_data)
        test_processed_data  = normalize_text(test_data)

        # Save the processed data
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
