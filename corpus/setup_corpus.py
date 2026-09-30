import nltk

# Define your custom directory path
custom_dir = "collageproject/corpus/"

# Add it to NLTK's search path
nltk.data.path.append(custom_dir)
nltk.download('punkt', download_dir=custom_dir)
nltk.download('stopwords', download_dir=custom_dir)
nltk.download('punkt_tab', download_dir=custom_dir)