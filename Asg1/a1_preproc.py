import sys
import argparse
import os
import json
import re
import string
import spacy
import html

indir = '/u/cs401/A1/data/';
required_files_dir = 'datasets/reddit/'
student_id = 0
EMPTY_STRING = ""
SPACE = " "
NEWLINE_REGEX = r'\n'
nlp = spacy.load('en', disable=['parser', 'ner'])

# Characters taken from https://perishablepress.com/stop-using-unsafe-characters-in-urls/
URL_REGEX = r'(http:\/\/|www\.)[$.+!*,;?:@=&\w\-\'\(\)\/]*'  
abbrev = []
stopwords = []

def detect_abbrev(pattern):
    if pattern.group(0).strip() in abbrev:
        return pattern.group(0)
    else:
        return pattern.group(1) + SPACE + "." + SPACE

def detect_stopwords(pattern):
    if pattern.group(1).strip() in stopwords:
        return ""
    else:
        return pattern.group(0)

def detect_eos(line, is_tokenized):
    if (is_tokenized):
        line = re.sub(r"(\S+)/(\S+)", r"\1", line)
    
    # Newline after EOS punctuations
    line = re.sub(r"(\w) (\.|!|\?|;|:|\") (\w)", r"\1 \2 \n \3", line)

    # Move EOS if punctuations followed by double quotations
    line = re.sub(r" (\.|!|?) \" ", r" \1 \" \n ", line)

    # Disqualify boundary with ? or ! if followed by a lowercase letter
    line = re.sub(r"(?|!) \n ([a-z])", r"\1 \2", line)
 
def tokenize(line):
    if ('/' in line):
        print("Backslash detected in line =>", line)

    tokenize_line = ""
    doc = nlp(line)
    for token in doc:
        tokenize_line = tokenize_line + token.text + "/" + token.tag_ + SPACE
    return tokenize_line.strip()

def lemmatize(line, is_tokenized):
    lemmatize_line = ""
    
    if (is_tokenized):
        line = re.sub(r"(\S+)/(\S+)", r"\1", line)
    
    doc = nlp(line)
    for token in doc:
        if (token.lemma_.startswith("-")):
            lemmatize_line = lemmatize_line + token.text + "/" + token.tag_ + SPACE
        else:
            lemmatize_line = lemmatize_line + token.lemma_ + "/" + token.tag_ + SPACE

    return lemmatize_line.strip()

def preproc1(comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''
    if (comment == "[deleted]"):
        return EMPTY_STRING

    modComm = comment
    if 1 in steps:
        modComm = re.sub(NEWLINE_REGEX, SPACE, modComm)
    if 2 in steps:
        modComm = html.unescape(modComm)
    if 3 in steps:
        modComm = re.sub(URL_REGEX, EMPTY_STRING, modComm)
    if 4 in steps:
        # Replace ." ?" !" as . " ? " ! "
        modComm = re.sub(r"(\w)(!|\.|\?)[\s]*(\")", r"\1 \2 \3", modComm)
        
        # Tokenize grammatically incorrect multiple punctuations as a single token  (breaks grammatically incorrect <dogs'.>)
        modComm = re.sub(r"(([^\s\w]{2,})(\")|([^\s\w]{2,}))", r" \2\4 \3 ", modComm)

        # Tokenize other punctuations :;[]{}()<>!?, coming after a character
        modComm = re.sub(r"(\w)([:;<>,?!\"\[\]\{\}\(\)]){1}", r"\1 \2 ", modComm)

        # Tokenize other punctuations :;<>,?!"[]{}() coming before a character
        modComm = re.sub(r"([:;<>,?!\"\[\]\{\}\(\)]){1}(\w)", r" \1 \2", modComm)

        # Tokenize period ignoring abbreviations
        modComm = re.sub(r"(\w[\w\.]*\w)[\.] ", detect_abbrev, modComm)

    if 5 in steps:
        modComm = re.sub(r"(\w)(n't|'s|'ve|'m|'re|'ll)", r"\1 \2 ", modComm)
        modComm = re.sub(r"s' ", r"s ' ", modComm)
    if 6 in steps:
        modComm = re.sub(r"\s+", r" ", modComm)
        modComm = tokenize(modComm)
    if 7 in steps:
        # Case sensitive in searching for stopwords
        modComm = re.sub(r"(\w+)/(\w+)", detect_stopwords, modComm)
    if 8 in steps:
        modComm = re.sub(r"\s+", r" ", modComm)
        modComm = lemmatize(modComm, 6 in steps)
    if 9 in steps:
        modComm = re.sub(r"\s+", r" ", modComm)
        modComm = detect_eos(modComm, 6 in steps or 8 in steps)
        # modComm = re.sub(r" (\.|!|?|;|:|\") ", r" \1 \n ", modComm)
        print("TODO")
    if 10 in steps:
        modComm = re.sub(r"(\S+)/(\S+)", lambda pattern: pattern.group(1).lower() + "/" + pattern.group(2), modComm)
        
        
    return modComm

def preproc(data, category):
    for i in range(len(data)):
        json_line = json.loads(data[i])
        preproc_body = preproc1(json_line['body'])
    
        json_line['body'] = preproc_body
        json_line['cat'] = category

        data[i] = json.JSONEncoder().encode(json_line)
    
    return data

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for filename in files:
            fullFile = os.path.join(subdir, filename)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            
            start_index = student_id % len(data)
            if (len(data) - (start_index + 1) < args.max):
                data = data[start_index : ] + data[0: args.max - len(data) + start_index]
            else:
                data = data[start_index : start_index + args.max]
            
            preproc_data = preproc(data, filename)
            allOutput = allOutput + preproc_data

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'filename' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'
            
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='s')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000, type=int)
    parser.add_argument("-l", "--local", help="Specify the flag when running locally", action="store_true")
    args = parser.parse_args()

    student_id = args.ID[0]
    abbrev = open(required_files_dir + "wordlists/abbrev.english").read().splitlines()
    stopwords = open(required_files_dir + "wordlists/StopWords").read().splitlines()

    # Changes input data directory when running in a local system
    if (args.local):
        indir = "./datasets/reddit/data/"

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)
        
    main(args)
