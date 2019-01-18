import re
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])
required_files_dir = "datasets/reddit/"
abbrev = open(required_files_dir + "wordlists/abbrev.english").read().splitlines()
stopwords = open(required_files_dir + "wordlists/StopWords").read().splitlines()
SPACE = " "
def detect_abbrev(pattern):
        if pattern.group(0).strip() in abbrev:
            return pattern.group(0)
        else:
            return pattern.group(1) + " " + "." + " "

def detect_stopwords(pattern):
    # import pdb; pdb.set_trace()
    if pattern.group(1).strip().lower() in stopwords:
        return ""
    else:
        return pattern.group(0)

def detect_eos(line, is_tokenized):
    if (is_tokenized):
        line = re.sub(r"(\S+)/(\S+)", r"\1", line)

    # Newline after EOS punctuations
    line = re.sub(r"(\w) ([\.!\?;:\"]+) (\w)", r"\1 \2 \n\3", line)

    # Move EOS if punctuations followed by double quotations
    line = re.sub(r" (\.|!|\?) \" ", r" \1 \" \n", line)

    # Disqualify boundary with ? or ! if followed by a lowercase letter
    line = re.sub(r"(\?|!|\.) \n([a-z])", r"\1 \2", line)

    return line

modComm = "How was your journey?! I am going to or gonna do Torontos' and enjoy night-life. \"Hi I am Dr. cm. Sumeet.\"\"How are you? I'm great!!!\" hahaha that's grest... great!!! I couldn't. I'm courses, $10,110 and 'll let you know"
# modComm = "\"What is it?\", remarked Mac, \"Is she okay?\" I am gonna"
        # Replace ." ?" !" as . " ? " ! "
modComm = re.sub(r"(\w)(!|\.|\?)[\s]*(\")", r"\1 \2 \3", modComm)
modComm = re.sub(r"\", ", "\" , ", modComm)
# Tokenize grammatically incorrect multiple punctuations as a single token  (breaks grammatically incorrect <dogs'.>)
modComm = re.sub(r"(([^\s\w]{2,})(\")|([^\s\w]{2,}))", r" \2\4 \3 ", modComm)

# Tokenize other punctuations :;[]{}()<>!?, coming after a character
modComm = re.sub(r"(\w)([:;<>?!\"\[\]\{\}\(\)]){1}", r"\1 \2 ", modComm)

# Tokenize other punctuations :;<>,?!"[]{}() coming before a character
modComm = re.sub(r"([:;<>?!\"\[\]\{\}\(\)]){1}(\w)", r" \1 \2", modComm)

# Tokenize comma coming before a character (comma handled separately to prevent from splitting numbers, eg: 10,000)
modeComm = re.sub(r",(a-ZA-Z_)", r", \1", modComm) 

# Tokenize comma coming after a character
modeComm = re.sub(r"(a-ZA-Z_),", r"\1 ,", modComm) 

# Tokenize period ignoring abbreviations
modComm = re.sub(r"(\w[\w\.]*\w)[\.] ", detect_abbrev, modComm)


modComm = re.sub(r"(\w[\w\.]*\w)[\.] ", detect_abbrev, modComm)
modComm = re.sub(r"(\w)(n't|'d|'ve|'m|'re|'ll|'n)", r"\1 \2 ", modComm)
modComm = re.sub(r"s' ", r"s ' ", modComm)

modComm = re.sub(r"\s+", " ", modComm)

print(modComm)
doc = spacy.tokens.Doc(nlp.vocab, words=modComm.strip().split(" "))
doc = nlp.tagger(doc)
temp = ""
for token in doc:
    temp = temp + token.text + "/" + token.tag_ + " "
# print("Tokenised!")
print(temp)
modComm = re.sub(r"(\S+)/(\S+)", detect_stopwords, temp)
# print("Wthout stopwords")
# print(modComm)

modComm = re.sub(r"\s+", r" ", modComm)
temp = ""
modComm = re.sub(r"(\S+)/(\S+)", r"\1", modComm)
doc = spacy.tokens.Doc(nlp.vocab, words=modComm.strip().split(" "))
doc = nlp.tagger(doc)
for token in doc:
    if (token.lemma_.startswith("-")):
        temp = temp + token.text + "/" + token.tag_ + " "
    else:
        temp = temp + token.lemma_ + "/" + token.tag_ + " "
modComm = temp.strip()
modComm = re.sub(r"\s+", SPACE, modComm)
modComm = detect_eos(modComm, True)
print(modComm)
re.sub(r"(\S+)/(\S+)", lambda pattern: pattern.group(1).lower() + "/" + pattern.group(2), temp)
print(len(re.compile(r"[^\s\w]*\w[\S]*").findall(modComm)))
# print(modComm)
