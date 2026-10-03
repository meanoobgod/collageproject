import os
import re
import nltk

# Calculate absolute path to corpus relative to Engine.py
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.abspath(os.path.join(APP_DIR, "..", "corpus", "collageproject", "corpus"))

# Force NLTK to look in your custom directory first
if CORPUS_DIR not in nltk.data.path:
    nltk.data.path.insert(0, CORPUS_DIR)

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from rake_nltk import Rake


class Engine:
    def __init__(self, LoggingLevel=True):
        self.LoggingLevel = LoggingLevel
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

        # Load Stopwords & Stemmer
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', cleaned).strip()

    def get_embeddings(self, texts: list) -> np.ndarray:
        """Embeds multiple texts at once and returns normalized L2 vectors."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        encoded = self.tokenizer.encode_batch([text.lower() for text in texts])
        input_ids = np.asarray([x.ids for x in encoded], dtype=np.int64)
        attention_mask = np.asarray([x.attention_mask for x in encoded], dtype=np.int64)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        if "token_type_ids" in self.input_names:
            inputs["token_type_ids"] = np.asarray([x.type_ids for x in encoded], dtype=np.int64)

        outputs = self.session.run(None, inputs)
        token_embeddings = outputs[0]  # [batch, tokens, hidden_size]

        mask = attention_mask[..., None].astype(np.float32)
        summed = np.sum(token_embeddings * mask, axis=1)
        counts = np.sum(mask, axis=1)

        embedding = summed / np.maximum(counts, 1e-9)
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / np.maximum(norms, 1e-12)

        return embedding.astype(np.float32)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.get_embeddings([text])[0]

    def getScore(self, a: str, b: str) -> float:
        vec_a, vec_b = self.get_embeddings([a.lower(), b.lower()])
        return float(np.dot(vec_a, vec_b))

    def getKeywords(self, parent: str) -> list:
        """Extracts stemmed keywords filtered by Nouns, Verbs, and Adjectives using POS Tagging."""
        tokens = word_tokenize(parent.replace('\n', ' ').lower())
        clean_tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words]

        # Part-of-speech tagging to filter out filler words
        try:
            tagged = nltk.pos_tag(clean_tokens)
            valid_pos = ('NN', 'NNS', 'NNP', 'JJ', 'VB', 'VBG', 'VBD', 'VBN')
            filtered_words = [word for word, tag in tagged if tag.startswith(valid_pos)]
        except Exception:
            filtered_words = clean_tokens  # Fallback if POS tagger is missing resource

        keywords = [self.stemmer.stem(word) for word in filtered_words]
        return list(dict.fromkeys(keywords))  # Deduplicate keeping order

    def getKeyphrases(self, parent: str) -> list:
        """Extracts keyphrases from text using RAKE."""
        RakeSearch = Rake()
        RakeSearch.extract_keywords_from_text(parent)
        phrases = list(set([x for x in RakeSearch.get_ranked_phrases() if len(x) > 4 and len(x.split()) >= 2]))

        if self.LoggingLevel:
            print(f"[Engine.getKeyphrases] Extracted phrases: {phrases}")

        return phrases

    def detect_negation_penalty(self, student_text: str, keywords: list, window_size: int = 4) -> float:
        """Checks if a negation token occurs near key domain terms and applies a penalty factor."""
        negation_tokens = {"not", "no", "never", "cannot", "n't", "lacks", "without", "neither", "nor", "false"}
        tokens = word_tokenize(student_text.lower())

        for i, token in enumerate(tokens):
            if token in negation_tokens:
                window = tokens[i + 1 : i + 1 + window_size]
                stemmed_window = [self.stemmer.stem(w) for w in window if w.isalpha()]

                if any(kw in stemmed_window for kw in keywords):
                    if self.LoggingLevel:
                        print(f"[Negation Penalty Applied] Negation '{token}' near keyword in: {' '.join(window)}")
                    return 0.25  # Deduct 75% score for negated keyword usage

        return 1.0  # No penalty

    def searchKeywords(self, student_text: str, keywords_to_search: list) -> float:
        """Proportional keyword search with smooth thresholding."""
        if not keywords_to_search:
            return 1.0

        tokens = word_tokenize(student_text.replace('\n', ' ').lower())
        token_set = set(self.stemmer.stem(word) for word in tokens if word.isalpha())

        total_matches = sum(1 for kw in keywords_to_search if kw in token_set)
        match_ratio = total_matches / len(keywords_to_search)

        if self.LoggingLevel:
            print(f"[Engine.searchKeywords] Keyword match ratio: {match_ratio:.2f}")

        # Smooth linear scaling (Target target = 60% match for full credit)
        target_threshold = 0.60
        return min(1.0, match_ratio / target_threshold)

    def searchKeyPhrase(self, student_text: str, keyphrase_to_search: list) -> float:
        """Compares teacher keyphrases against student sentences for semantic alignment."""
        if not keyphrase_to_search:
            return 1.0

        student_sentences = sent_tokenize(student_text)
        if not student_sentences:
            return 0.0

        all_texts = keyphrase_to_search + student_sentences
        embeddings = self.get_embeddings(all_texts)

        kp_count = len(keyphrase_to_search)
        kp_embeddings = embeddings[:kp_count]
        sent_embeddings = embeddings[kp_count:]

        # Cosine similarity matrix: [num_keyphrases, num_sentences]
        similarities = kp_embeddings @ sent_embeddings.T
        best_matches = np.max(similarities, axis=1)

        # Proportional keyphrase score (Threshold set at 0.50 similarity)
        valid_scores = [max(0.0, (score - 0.50) / 0.35) for score in best_matches]
        final_score = float(np.mean(valid_scores))

        if self.LoggingLevel:
            print(f"[Engine.searchKeyPhrase] Keyphrase match score: {final_score:.2f}")

        return min(1.0, final_score)

    def MarkQuestion(self, total_marks: int | float, teacher_answer: str, student_answer: str) -> float:
        """Main grading pipeline combining semantic similarity, phrase matching, and penalty filters."""
        if not student_answer.strip():
            return 0.0

        teacher_words = teacher_answer.replace("\n", " ").split()
        student_words = student_answer.replace("\n", " ").split()

        keywords = self.getKeywords(teacher_answer)
        cleaned_keywords = [self.clean_text(k) for k in keywords if k]

        # 1. Short Answer Logic (<= 25 words threshold)
        if len(teacher_words) <= 25:
            if self.LoggingLevel:
                print("[Short Answer Evaluation]")

            sim_score = self.getScore(teacher_answer, student_answer)
            negation_factor = self.detect_negation_penalty(student_answer, cleaned_keywords)

            # Linear scaling between 0.50 (0%) and 0.85 (100%) similarity
            if sim_score >= 0.85:
                multiplier = 1.0
            elif sim_score <= 0.50:
                multiplier = 0.0
            else:
                multiplier = (sim_score - 0.50) / (0.85 - 0.50)

            return round(total_marks * multiplier * negation_factor, 2)

        # 2. Long Answer Evaluation
        kws = self.searchKeywords(student_answer, cleaned_keywords)

        keyphrases = self.getKeyphrases(teacher_answer)
        cleaned_keyphrases = [self.clean_text(kp) for kp in keyphrases if kp]

        if cleaned_keyphrases:
            kps = self.searchKeyPhrase(student_answer, cleaned_keyphrases)
            base_score = (kws + kps) / 2.0
        else:
            base_score = kws

        # 3. Apply Penalties (Negation & Verbosity/Dictionary Attack)
        negation_factor = self.detect_negation_penalty(student_answer, cleaned_keywords)

        length_penalty = 1.0
        if len(student_words) > (len(teacher_words) * 3):
            excess_ratio = (len(student_words) / len(teacher_words)) - 3
            length_penalty = max(0.2, 1.0 - (excess_ratio * 0.1))
            if self.LoggingLevel:
                print(f"[Length Penalty Applied] Factor: {length_penalty:.2f}")

        final_score = total_marks * base_score * negation_factor * length_penalty
        return round(float(final_score), 2)
