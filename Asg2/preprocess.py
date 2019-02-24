import re

def preprocess(in_sentence, language):
    """ 
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation 

    INPUTS:
    in_sentence : (string) the original sentence to be processed
    language	: (string) either 'e' (English) or 'f' (French)
                    Language of in_sentence
                    
    OUTPUT:
    out_sentence: (string) the modified sentence
    """
    # TODO: Implement Function
    out_sentence = "SENTSTART "
    SPACE = " "

    # Separates EOS punctuations, commas, semicolons, colons, mathematical operators, quotation marks, dashes, parentheses
    in_sentence = re.sub(r"(!|\?|\.|,|:|;|\(|\)|\+|-|<|>|=|\")", r" \1 ", in_sentence)
    
    # Converts tokens to lowercase
    in_sentence = re.sub(r"(\S+)", lambda pattern: pattern.group(1).lower(), in_sentence)

    if language == 'f':
        in_sentence = re.sub(r"( |^)(l|qu)\'(\w)", r"\1\2' \3", in_sentence)
        in_sentence = re.sub(r"(\w)\'(on|il)( |$)", r"\1 '\2\3", in_sentence)
        in_sentence = re.sub(r"( |^)(b|c|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z)\'(\w+)", r"\1\2' \3", in_sentence)
        in_sentence = re.sub(r"( |^)d\'(?!(abord|accord|ailleurs|habitude)( |$))(\w+)", r"\1d' \4", in_sentence)

    out_sentence = out_sentence + in_sentence.strip() + " SENTEND"
    return out_sentence