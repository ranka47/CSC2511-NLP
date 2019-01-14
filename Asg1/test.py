import re
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])
def detect_abbrev(pattern):
        if pattern.group(0).strip() in ["Dr.", "cm."]:
            return pattern.group(0)
        else:
            return pattern.group(1) + " " + "." + " "

def detect_stopwords(pattern):
    if pattern.group(1).strip() in ["am", "are"]:
        return ""
    else:
        return pattern.group(0)


modComm = "Welcome. To Torontos'. \"Hi I am Dr. cm. Sumeet.\"\"How are you? I'm great!!!\" hahaha that's grest... its great!!! I couldn't."
# modComm = "Hi hkhkk!!!\"'"
modComm = re.sub(r"(\w)(!|\.|\?)[\s]*(\")", r"\1 \2 \3", modComm)

# Tokenise grammatically incorrect multiple punctuations as a single token
modComm = re.sub(r"(([^\s\w]{2,})(\")|([^\s\w]{2,}))", r" \2\4 \3 ", modComm)
# Tokenise other punctuations :;[]{}()<>!?, coming after a character
modComm = re.sub(r"(\w)([:;<>,?!\"\[\]\{\}\(\)]){1}", r"\1 \2 ", modComm)

# Tokenise other punctuations :;<>,?!"[]{}() coming after a character
modComm = re.sub(r"([:;<>,?!\"\[\]\{\}\(\)]){1}(\w)", r" \1 \2", modComm)

# Tokenise period ignoring abbreviations
modComm = re.sub(r"(\w[\w.]*\w)[\.] ", detect_abbrev, modComm)

modComm = re.sub(r"(\w)(n't|'s|'ve|'m)", r"\1 \2 ", modComm)

modComm = re.sub(r"s' ", r"s ' ", modComm)
modComm = re.sub(r"\s+", r" ", modComm)

print(modComm)

doc = nlp(modComm)
temp = ""
for token in doc:
    temp = temp + token.text + "/" + token.tag_ + " "
print("Tokenised!")
print(temp)
modComm = re.sub(r"(\w+)/(\w+)", detect_stopwords, temp)
print("Wthout stopwords")
print(modComm)

modComm = re.sub(r"\s+", r" ", modComm)
temp = ""
modComm = re.sub(r"(\S+)/(\S+)", r"\1", modComm)
doc = nlp(modComm)
for token in doc:
    if (token.lemma_.startswith("-")):
        temp = temp + token.text + "/" + token.tag_ + " "
    else:
        temp = temp + token.lemma_ + "/" + token.tag_ + " "
print(temp)
modComm = re.sub(r" (.|!|\?|;|:|\") ", r" \1 \n ", modComm)
re.sub(r"(\S+)/(\S+)", lambda pattern: pattern.group(1).lower() + "/" + pattern.group(2), temp)
print(modComm)
