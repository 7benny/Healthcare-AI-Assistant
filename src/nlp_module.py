# nlp_module.py

import spacy
from spacy.matcher import PhraseMatcher
import joblib

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load the symptom list
symptom_list = joblib.load('symptom_list.pkl')

# Initialize the PhraseMatcher
phrase_matcher = PhraseMatcher(nlp.vocab)
patterns = [nlp.make_doc(symptom.lower()) for symptom in symptom_list]
phrase_matcher.add('SYMPTOMS', None, *patterns)

def extract_symptoms(user_input):
    doc = nlp(user_input.lower())
    matches = phrase_matcher(doc)
    extracted_symptoms = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        extracted_symptoms.add(span.text)
    return list(extracted_symptoms)

if __name__ == '__main__':
    user_input = "I have a fever and cough with a sore throat"
    symptoms = extract_symptoms(user_input)
    print("Extracted Symptoms:", symptoms)
