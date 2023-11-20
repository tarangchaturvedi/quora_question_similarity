import gensim.downloader as api
import joblib
import numpy as np
import distance

def load_word2vec_model():
    return api.load("glove-wiki-gigaword-100")

def word_to_vec(text, word2vec_model):
    words = text.split()
    word_vectors = [word2vec_model[word] if word in word2vec_model else np.zeros(word2vec_model.vector_size) for word in words]
    avg_vecs = np.mean(word_vectors, axis=0)
    return avg_vecs

def create_word2vec_features(df1):
    df = df1.copy()
    word2vec_model = load_word2vec_model()
    joblib.dump(word2vec_model, 'word2vec_model.joblib')
    df['q1_feats'] = df['question1'].apply(lambda text: word_to_vec(text, word2vec_model))
    df['q2_feats'] = df['question2'].apply(lambda text: word_to_vec(text, word2vec_model))
    df.fillna('', inplace = True)
    return df

def normalized_word_Common(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)
        
def normalized_word_Total(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * (len(w1) + len(w2))

def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

# get the Longest Common sub string
def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

from sklearn.preprocessing import MinMaxScaler

def standardize_data(data):
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    joblib.dump(scaler, 'MinMaxScaler.joblib')    
    # Fit the scaler to the data and transform the data
    scaled_data = scaler.fit_transform(data)

    return scaled_data
