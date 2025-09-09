import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "qwiklabs-gcp-02-d8cd54229bbe"

import yaml
import logging
import google.cloud.logging
from flask import Flask, render_template, request


from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
PROJECT_ID = "qwiklabs-gcp-02-d8cd54229bbe"
LOCATION = "us-central1"  # Vertex AI GenAI+Embeddings commonly supported here

# 1) Strongly recommended: initialize Vertex AI explicitly
vertexai.init(project=PROJECT_ID, location=LOCATION)
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "FreshBot"
SUBTITLE = "Your Friendly Restaurant Safety Expert"

app = Flask(__name__)

# Initializing the Firebase client
db = firestore.Client()

# TODO: Instantiate a collection reference
collection = db.collection("food-safety")

# TODO: Instantiate an embedding model here
from langchain_google_vertexai import VertexAIEmbeddings
eembedding_model = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=PROJECT_ID,  # explicit, to avoid ADC ambiguity
)

# TODO: Instantiate a Generative AI model here
from langchain_google_vertexai import VertexAI
gen_model = VertexAI(
    model_name="gemini-2.0-flash",
    temperature=0,
    project=PROJECT_ID,
)

# TODO: Implement this function to return relevant context
# from your vector database
def search_vector_database(query: str):
    query_embedding = embedding_model.embed_query(query)
    query_vector = Vector(query_embedding)
    docs = collection.find_nearest(
        "embedding",
        query_vector=query_vector,
        distance_measure=DistanceMeasure.DOT_PRODUCT,
        limit=5
    ).get()
    
    pieces = []
    for doc in docs:
        data = doc.to_dict()
        if "content" in data:
            pieces.append(data["content"])
    
    context = "\n".join(pieces)
    return context

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
def ask_gemini(question):
    context = search_vector_database(question)
    # Prepare the prompt for Gemini
    prompt =f"""Use the following food safety manual context to answer the question.

Context:
{context_text}

Question:
{question}
"""
    # Set safety settings: block only high-probability dangerous content
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
        )
    ]

    # Generate the answer
    response = gen_model.generate_content(
        prompt,
        safety_settings=safety_settings
    )

    
    # 1. Generate the embedding of the query

    # 2. Get the 5 nearest neighbors from your collection.
    # Call the get() method on the result of your call to
    # find_neighbors to retrieve document snapshots.

    # 3. Call to_dict() on each snapshot to load its data.
    # Combine the snapshots into a single string named context


    # Don't delete this logging statement.
    logging.info(
        context, extra={"labels": {"service": "cymbal-service", "component": "context"}}
    )
    return context

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm FreshBot, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "cymbal-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
