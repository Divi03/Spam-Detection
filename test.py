import pickle

# Load the model from the Pickle file
with open('models/vectorizer.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Inspect the loaded model object
print(loaded_model)
