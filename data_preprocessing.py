import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from nltk.stem import PorterStemmer
import re
from bs4 import BeautifulSoup


def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(keep='first', inplace=True)
    df.fillna('', inplace = True)

    # preprocessing each question
    df["question1"] = df["question1"].apply(preprocess)
    df["question2"] = df["question2"].apply(preprocess)

    df.fillna('', inplace = True)
    return df

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("'", "'").replace("'", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
        
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
                   
    return x
