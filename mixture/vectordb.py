import chromadb

# Connect to the ChromaDB database
db = chromadb.connect("localhost", 8080)

# Create a new collection
collection = db.create_collection("my_collection")

# Define the schema for the collection
schema = {
    "id": chromadb.String(),
    "features": chromadb.Array(chromadb.Float())
}

# Create the collection with the specified schema
collection.create(schema)
