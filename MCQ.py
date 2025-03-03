import random
import re
import nltk
import pke
import string
import requests
import json
from flashtext import KeywordProcessor
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords, wordnet as wn
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

# Load text file
file = open("article.txt", "r", encoding="utf-8")
text = file.read()
file.close()

def get_important_words(art):
    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=art, language='en')
    pos = {'NOUN', 'PROPN'}
    stops = stopwords.words('english') + list(string.punctuation)
    extractor.candidate_selection(pos=pos)
    extractor.candidates = {key: val for key, val in extractor.candidates.items() if key.lower() not in stops}
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=25)
    return [phrase[0] for phrase in keyphrases]

def split_text_to_sents(art):
    s = sent_tokenize(art)
    return [sent.strip() for sent in s if len(sent) > 15]

def map_sents(imp_words, sents):
    processor = KeywordProcessor()
    key_sents = {word: [] for word in imp_words}
    for word in imp_words:
        processor.add_keyword(word)
    for sent in sents:
        found = processor.extract_keywords(sent)
        for each in found:
            key_sents[each].append(sent)
    for key in key_sents:
        key_sents[key] = sorted(key_sents[key], key=len, reverse=True)
    return key_sents

def get_word_sense(sent, word):
    word = word.lower().replace(" ", "_")
    synsets = wn.synsets(word, 'n')

    if not synsets:
        return None  # No synsets found

    try:
        wup = max_similarity(sent, word, 'wup', pos='n') if synsets else None
    except IndexError:
        wup = None  # Fix: Prevent IndexError if max_similarity() fails

    try:
        adapted_lesk_output = adapted_lesk(sent, word, pos='n') if synsets else None
    except IndexError:
        adapted_lesk_output = None  # Fix: Prevent IndexError if adapted_lesk() fails

    if wup and adapted_lesk_output:
        return synsets[min(synsets.index(wup), synsets.index(adapted_lesk_output))]
    elif wup:
        return wup
    elif adapted_lesk_output:
        return adapted_lesk_output

    return synsets[0]  # Default to the first synset if others fail

def get_distractors(syn, word):
    dists = []
    word = word.lower().replace(" ", "_")
    hypernym = syn.hypernyms() if syn else []
    if not hypernym:
        return dists
    for each in hypernym[0].hyponyms():
        name = each.lemmas()[0].name().replace("_", " ")
        if name.lower() != word and name not in dists:
            dists.append(name.title())
    return dists

def get_distractors2(word):
    word = word.lower().replace(" ", "_")
    dists = []
    url = f"http://api.conceptnet.io/query?node=/c/en/{word}/n&rel=/r/PartOf&start=/c/en/{word}&limit=5"
    obj = requests.get(url).json()
    for edge in obj.get('edges', []):
        link = edge['end']['term']
        url2 = f"http://api.conceptnet.io/query?node={link}&rel=/r/PartOf&end={link}&limit=10"
        obj2 = requests.get(url2).json()
        for edge in obj2.get('edges', []):
            word2 = edge['start']['label']
            if word2 not in dists and word.lower() not in word2.lower():
                dists.append(word2)
    return dists

def generate_mcqs(num_questions=5):
    imp_words = get_important_words(text)
    sents = split_text_to_sents(text)
    mapped_sents = map_sents(imp_words, sents)
    mapped_dists = {}

    for each in mapped_sents:
        if not mapped_sents[each]:
            continue

        wordsense = get_word_sense(mapped_sents[each][0], each)

        if wordsense:
            dists = get_distractors(wordsense, each)
        else:
            dists = get_distractors2(each)

        if dists:
            mapped_dists[each] = dists

    mcqs = []
    available_qs = [(k, v[0]) for k, v in mapped_sents.items() if v]

    if not available_qs:
        print("\nNo questions available.")
        return []

    selected_qs = random.sample(available_qs, min(num_questions, len(available_qs)))

    for i, (keyword, sent) in enumerate(selected_qs, start=1):
        p = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        question = p.sub("________", sent)
        options = [keyword.capitalize()] + mapped_dists.get(keyword, [])[:3]
        random.shuffle(options)
        mcqs.append((i, question, options))

    return mcqs

# Example usage
num_questions = int(input("Enter number of questions: "))
mcqs = generate_mcqs(num_questions)

if mcqs:
    print("\n************************************** Multiple Choice Questions **************************************\n")
    for i, question, options in mcqs:
        print(f"Question {i}: {question}")
        for j, opt in zip('abcd', options):
            print(f"   {j}) {opt}")
        print()

