from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import io
import base64


# Initialize the Flask app
app = Flask(__name__)

# Step 1: Load the 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

# Step 2: Build the TF-IDF matrix
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X = vectorizer.fit_transform(documents)

# Step 3: Apply SVD to reduce dimensionality (LSA)
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)

# Step 4: Define function to retrieve the top 5 documents based on cosine similarity
def retrieve_documents(query, X_reduced, documents, vectorizer, svd, top_n=5):
    query_tfidf = vectorizer.transform([query])
    query_reduced = svd.transform(query_tfidf)
    similarities = cosine_similarity(query_reduced, X_reduced).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]
    top_docs = [(documents[i], similarities[i]) for i in top_indices]
    
    return top_docs

# Routes for the web app
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        query = request.form['query']
        top_docs = retrieve_documents(query, X_reduced, documents, vectorizer, svd)
        
        # Prepare the documents and their similarity scores for rendering
        doc_texts = [doc for doc, sim in top_docs]
        doc_sims = [sim for doc, sim in top_docs]
        doc_ids = [f'Doc {i+1}' for i in range(len(top_docs))]

        '''
        # Create the vertical bar chart using Matplotlib
        plt.figure(figsize=(10, 6))
        plt.bar(doc_ids, doc_sims, color='skyblue')  # Vertical bars
        plt.xlabel('Document ID')
        plt.ylabel('Cosine Similarity')
        plt.title('Cosine Similarity of Top Documents')
        '''

        fig, ax = plt.subplots()

        for (sim, id) in zip(doc_sims, doc_ids):
            p = ax.bar(id, sim, color='skyblue')

            ax.bar_label(p, label_type='center')
        ax.set_xlabel('Document ID')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cosine Similarity of Top Documents')

        # Save the plot to a PNG image in memory
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', query=query, docs=zip(doc_texts, doc_sims), plot_url=plot_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)