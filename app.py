# Library imports
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
import joblib
import nltk  # Natural Language Toolkit
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords  # For removing stopwords
from nltk.stem import WordNetLemmatizer  # For lemmatizing words
import tensorflow as tf
from tensorflow.keras.models import load_model  # For loading the LSTM model
from config import Review
import pandas as pd
import io
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app object
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Which domains can access your API
    allow_credentials=True,   # Allow cookies and authentication headers
    allow_methods=["*"],     # Which HTTP methods are allowed (GET, POST, etc.)
    allow_headers=["*"]      # Which HTTP headers are allowed
)

# Download NLTK data files (if not already downloaded)
nltk.download('stopwords')
nltk.download('wordnet')

logging.debug("NLTK data files downloaded")

# Load the tokenizer
tokenizer = joblib.load('lstm_tokenizer.pkl')
logging.debug("Tokenizer loaded")

# Load the LSTM model
model = load_model('best_lstm_model.h5')
logging.debug("LSTM model loaded")

@app.get('/')
def index():
    logging.debug("Index endpoint called")
    return {'message': 'You detect fake reviews here!'}

def clean_text(text):
    logging.debug(f"Cleaning text: {text}")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    logging.debug(f"Cleaned text: {text}")
    return text

max_sequence_length = 100

@app.post('/predict')
def predict(review: Review):
    logging.debug(f"Predict endpoint called with review: {review}")
    try:
        review_text = review.dict()['review']
        logging.debug(f"Review text: {review_text}")
        clean_review = clean_text(review_text)
        review_sequence = tokenizer.texts_to_sequences([clean_review])
        review_padded = tf.keras.preprocessing.sequence.pad_sequences(
            review_sequence, 
            maxlen=max_sequence_length, 
            padding='post'
        )
        logging.debug(f"Padded review sequence: {review_padded}")
        
        # Get prediction and convert to scalar
        prediction = float(model.predict(review_padded)[0][0])
        logging.debug(f"Prediction: {prediction}")
        
        return {
            'prediction': 'Fake review' if prediction > 0.5 else 'Real review',
            'confidence': round(prediction * 100, 2)
        }
    except Exception as e:
        logging.error(f"Error processing review: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing review: {str(e)}"
        )

@app.post("/process-reviews")
async def process_reviews_csv(file: UploadFile = File(...)):
    logging.debug("Process reviews CSV endpoint called")
    try:
        # Read CSV content
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        logging.debug("CSV content read into DataFrame")
        
        # Validate CSV structure
        if not all(col in df.columns for col in ['productId', 'review']):
            logging.error("CSV must contain 'product_id' and 'review' columns")
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'product_id' and 'review' columns"
            )
        
        # Process each review
        results = []
        for _, row in df.iterrows():
            logging.debug(f"Processing review for product ID: {row['productId']}")
            clean_review = clean_text(row['review'])
            review_sequence = tokenizer.texts_to_sequences([clean_review])
            review_padded = tf.keras.preprocessing.sequence.pad_sequences(
                review_sequence,
                maxlen=100,  # Make sure this matches your model's input size
                padding='post'
            )
            logging.debug(f"Padded review sequence: {review_padded}")
            
            # Get prediction
            prediction = float(model.predict(review_padded)[0][0])
            logging.debug(f"Prediction: {prediction}")
            
            # If review is predicted as real (prediction <= 0.5)
            if prediction <= 0.5:
                results.append({
                    'productId': row['productId'],
                    'review': row['review'],
                    'confidence': round((1 - prediction) * 100, 2)
                })
        
        logging.debug(f"Total reviews processed: {len(df)}, Real reviews count: {len(results)}")
        return {
            'real_reviews': results,
            'total_reviews': len(df),
            'real_reviews_count': len(results)
        }
        
    except Exception as e:
        logging.error(f"Error processing CSV file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CSV file: {str(e)}"
        )

if __name__ == '__main__':
    logging.debug("Starting the application")
    uvicorn.run(app, host='127.0.0.1', port=8000)

# Run the app with the following command:
# uvicorn app:app --reload
