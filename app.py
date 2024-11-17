from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pandas as pd
import http.client
import json
import base64
from io import BytesIO
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Initialize Flask app
app = Flask(__name__)

# Set up the FinBERT model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Function to fetch stock-related news
def search_stock(stock_name):
    formatted_stock_name = stock_name.replace(" ", "+") + "+stock"
    conn = http.client.HTTPSConnection("google-news13.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "8f5ac602b2msh867216cd233d33ap1dd359jsn8c2f49d7417a",
        'x-rapidapi-host': "google-news13.p.rapidapi.com"
    }
    conn.request("GET", f"/search?keyword={formatted_stock_name}&lr=en-US", headers=headers)
    res = conn.getresponse()
    data = res.read()
    return json.loads(data.decode("utf-8"))

# Convert timestamp to readable date and time
def convert_timestamp(timestamp):
    timestamp = int(timestamp) // 1000
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime('%d-%m-%Y')
    formatted_time = dt_object.strftime('%H:%M:%S')
    return formatted_date, formatted_time

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        stock_name = request.form['stock_name']
        news_data = search_stock(stock_name)

        titles = []
        sentiments = []
        for item in news_data["items"]:
            title = item["title"]
            sentiment = nlp(title)
            titles.append(title)
            sentiments.append(sentiment[0]['score'] if sentiment[0]['label'] == 'POSITIVE' else -sentiment[0]['score'])

        # Plot the sentiment values
        plt.bar(titles, sentiments, color=['green' if s > 0 else 'red' for s in sentiments])
        plt.xticks(rotation=90)
        plt.ylabel('Sentiment Score')
        plt.title(f'Sentiment Analysis for {stock_name}')
        plt.tight_layout()

        # Save the plot to a BytesIO buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        graph_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()

        # Pass the zip function explicitly
        return render_template('result.html', stock_name=stock_name, titles=titles, sentiments=sentiments, graph_url=graph_url, zip=zip)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
