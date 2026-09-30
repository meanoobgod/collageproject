import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake

class Engine():
    def __init__(self, LoggingLevel=False):
        self.LoggingLevel = LoggingLevel
        nltk.data.path.append('../corpus/collageproject/corpus')
        # Model paths relative to Engine.py
        APP_DIR = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(APP_DIR, "model")

        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        onnx_path = os.path.join(model_dir, "model.onnx")

        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Missing tokenizer at: {tokenizer_path}")

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"Missing ONNX model at: {onnx_path}")

        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # ONNX Session options
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1

        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_names = {x.name for x in self.session.get_inputs()}

        # Load Stopwords & Stemmer directly (NLTK_DATA handles path lookup)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def get_embeddings(self, texts: list) -> np.ndarray:
        """
            This function embeds multiple texts at once and returns
            normalized vectors.
        """

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        encoded = self.tokenizer.encode_batch([text.lower() for text in texts])

        input_ids = np.asarray([x.ids for x in encoded],dtype=np.int64)

        attention_mask = np.asarray([x.attention_mask for x in encoded],dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Some BERT-style models require token_type_ids
        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.asarray([x.type_ids for x in encoded], dtype=np.int64)

        outputs = self.session.run(None, inputs)

        # [batch, tokens, hidden_size]
        token_embeddings = outputs[0]

        # [batch, tokens, 1]
        mask = attention_mask[..., None].astype(np.float32)

        # Mean pooling over non-padding tokens
        summed = np.sum(token_embeddings * mask, axis=1)

        counts = np.sum(mask, axis=1)

        embedding = summed / np.maximum(counts, 1e-9)

        # L2 normalize entire batch
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)

        embedding = embedding / np.maximum(norms, 1e-12)

        return embedding.astype(np.float32)


    
    def get_embedding(self, text: str) -> np.ndarray:
        """
            This function embeddes the text to a vector, and retuns it.
        """

        return self.get_embeddings([text])[0]


    
    def searchKeywords(self, parent: str, keywords_to_search: list) -> float:
        # <--- LOWERCASE AND STEM ALL TOKENS IN THE PARENT TEXT
        tokens = word_tokenize(parent.replace('\n', ' ').lower())
        token_set = set(self.stemmer.stem(word) for word in tokens if word.isalpha())
        
        total = 0
        for kw in keywords_to_search:
            if kw in token_set:
                total += 1

        if not keywords_to_search:
            return 0

        if self.LogginfLevel:
            print(f"[Engine.searchKewords] % of keywords matches: {total / len(keywords_to_search)}")

        if total / len(keywords_to_search) >= 0.40: #change this to change the strictness of the checking
            return 1
        elif total / len(keywords_to_search) >= 0.25:
            return 0.5
        else:
            return 0

    
    def searchKeyPhrase(self, parent: str, keyphrase_to_search: list) -> float:
        RakeSearch = Rake()
        sentences = RakeSearch.extract_keywords_from_text(parent)

        sentences = list(set([x for x in RakeSearch.get_ranked_phrases() if len(x) > 4 and len(x.split()) >= 2]))

        if not keyphrase_to_search or not sentences:
            return 0


        all_texts = keyphrase_to_search + sentences

        embeddings = self.get_embeddings(all_texts)

        keyphrase_count = len(keyphrase_to_search)

        keyphrase_embeddings = embeddings[:keyphrase_count]
        sentence_embeddings = embeddings[keyphrase_count:]

        # Calculate all cosine similarities at once
        similarities = keyphrase_embeddings @ sentence_embeddings.T

        # For every keyphrase, find its best matching sentence
        best_matches = np.max(similarities, axis=1)

        total = np.sum(best_matches >= 0.50)

        score = total / len(keyphrase_to_search)

        if self.LogginfLevel:
            print(f"[Engine.searchKeyPhrase] % of keyphrase matches: {score}")
            print("sentences: ", sentences)

        if score >= 0.30: #change this to change the strictness of the checking.
            return 1
        elif score >= 0.20:
            return 0.5
        else:
            return 0


    
    def getKeyphrases(self, parent: str) -> list:
        RakeSearch = Rake()
        _ = RakeSearch.extract_keywords_from_text(parent)

        sap = list(set([x for x in RakeSearch.get_ranked_phrases() if len(x) > 4 and len(x.split()) >= 2]))

        if self.LogginfLevel:
            print(f"[Engine.getKeyphrases] System accepect phrases: {sap}")

        return sap


    
    def getKeywords(self, parent: str) -> list:
        tokens = word_tokenize(parent.replace('\n', ' ').lower())

        return list(dict.fromkeys(self.stemmer.stem(word) for word in tokens if word.isalpha() and word not in self.stop_words))

    
    def getScore(self, a: str, b: str) -> float:
        vec_a, vec_b = self.get_embeddings([a.lower(), b.lower()])

        return float(np.dot(vec_a, vec_b))


    
    def MarkQuestion(self, total_marks: int | float,teacher_answer:str, student_answer: str):#, keywords: list, keyphrase: list = []):
        """
            How to score a text with the keywords and keyPhrase
            `Keyword`: single word simple fussy search will return the values, x% > full; y%> half else: 0
            `Keyphrase`: group of words, has to do a cosine search, x% > full, y% > half; else 0

            return the mean of both of the values.
        """
        if len(teacher_answer.replace("\n", " ").split(" ")) <= 200:
            if self.getScore(teacher_answer, student_answer) >= 0.80:
                return total_marks
            else:
                return 0
            
        keywords = self.getKeywords(teacher_answer)
        keyphrase = self.getKeyphrases(teacher_answer)
        #only have to check paper with keywords
        kws = self.searchKeywords(student_answer, keywords)

        if len(keyphrase) != 0:
            #has to check paper with keywords + keyphrase
            kps = self.searchKeyPhrase(student_answer, keyphrase)
            
            return total_marks * (kws + kps) / 2 # mean value of both the results

        return total_marks * kws # if len(parent < 50 words)
