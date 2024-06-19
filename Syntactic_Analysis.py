"""
## Installation Steps
- pip install -r requirements.txt
- python -m spacy download en_core_web_sm
- Install java, CoreNLP requires java

##############Run this one time to install corenlp server###########
# import stanza
# stanza.install_corenlp(dir="../corenlp_server") # Spacify path of directory where you want to install corenlp server
# stanza.download_corenlp_models(model='english-kbp', version='4.2.2', dir="../corenlp_server")

"""
from styleframe import StyleFrame
import pandas as pd
import numpy as np
import spacy
import stanza
from stanza.server import CoreNLPClient
stanza.download('en',processors="tokenize,mwt")       # This downloads the English models for the neural pipeline

nlp_stanza = stanza.Pipeline('en') # This sets up a default neural pipeline in English

nlp = spacy.load('en_core_web_sm')  # or whatever model you downloaded
# Read input file with specific sheetname
inputFile = "input.xlsx" # original file "grouped_output_gold_standard.xlsx" shared by vamshidhar
df = pd.read_excel(inputFile)

# Output file name
outputFile = "output.xlsx"

# start corenlp server
with CoreNLPClient(
        annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'],
        timeout=30000,
        endpoint='http://localhost:8000',
        be_quiet=True,
        memory='6G') as client:
    mwts = df["MWT"].dropna().to_list()
    # for column in df.columns:
    #     mwts = mwts + df[column].dropna().to_list()

    # uniques mwts
    unique_mwts = np.unique(np.array(mwts))

    # Rank the MWTs according to the alphabet
    sort_mwts = np.sort(unique_mwts)
    op_arr = []

    for i,mwt in enumerate(sort_mwts):
        merge_output = []
        alternative_pairs = []
        op_alternative_pairs = ''
        print(i)
        print("mwt => ",mwt)

        # spacy bigram
        doc = nlp(str(mwt))
        # replace space (" ") with underscore("_"), It will help to identify unique bigrams
        spacy_mwt = [f"{token.text}_{token.head}" for token in doc if str(token.text) != str(token.head)]
        
        # stanza
        doc_stanza = nlp_stanza(mwt)
        # replace space (" ") with underscore("_"), It will help to identify unique bigrams
        stanza_mwt = [f'{word.text}_{sent.words[word.head-1].text}' for sent in doc_stanza.sentences for word in sent.words if word.head > 0 and str(word.text) != str(sent.words[word.head-1].text)]
        # CoreNLP
        corenlp_mwt = []
        ann = client.annotate(mwt)
        offset = 0 # keeps track of token offset for each sentence
        for sentence in ann.sentence:
            # extract dependency parse
            dp = sentence.basicDependencies
            # build a helper dict to associate token index and label
            token_dict = {sentence.token[i].tokenEndIndex-offset : sentence.token[i].word for i in range(0, len(sentence.token))}
            offset += len(sentence.token)

            # build list of (source, target) pairs
            out_parse = [(dp.edge[i].source, dp.edge[i].target) for i in range(0, len(dp.edge))]
            corenlp_bigram = []
            for source, target in out_parse:
                if(str(token_dict[target]) != str(token_dict[source])):
                    # join with underscore to easily remove duplicates
                    corenlp_bigram.append(f"{token_dict[target]}_{token_dict[source]}")
            corenlp_mwt = corenlp_bigram
        
        # devide bigrams in to alternative and non-alternatice pairs
        
        # Rules to move bigram in alternative pairs:
            # 1) The bigrams that have the same first word but different second words (multilingual vocabulary, multilingual recognition), 
            # 2) The bigrams that are composed of the same words in reversed positions (multilingual recognition, recognition multilingual)
        
        # Rules to move bigram in non-alternative pairs:
            # 1) If a bigram is absent from the rows of alternative bigrams, it is included into the set of ‘non-alternative pairs’

        # merge all depedency pairs output
        merge_output = np.unique(spacy_mwt + stanza_mwt + corenlp_mwt).tolist()
        # array of each words of all bigrams
        merge_output_arr = [word.split("_") for word in merge_output]
        # array of first word of bigram, Alternative pairs Rule 1).
        # Identify type 1
        merge_output_first_word = np.unique([word.split("_")[0] for word in merge_output])
        alternative_pairs = []
        op_alternative_pairs = ''

        for unique_start_word in merge_output_first_word:
            type_1_terms = [bigram for bigram in merge_output if bigram.startswith(f"{unique_start_word}_")]
            if(len(type_1_terms) > 1):
                alternative_pairs.append(type_1_terms)

        # Identify type 2(Alternative Pair Rule 2. Bigram that are composed to same word)
        type_2_terms = []
        for bigram in merge_output_arr:
            for biram in merge_output:
                if((biram.startswith(f"{bigram[1]}_") and biram.endswith(f"_{bigram[0]}"))):
                    type_2_terms.append([biram, '_'.join(bigram)])

        unique_list_type_2_terms = []
        if(len(type_2_terms) > 0):   
            unique_set_type_2_terms = set(map(lambda x: tuple(sorted(x)),type_2_terms))
            unique_list_type_2_terms = [list(t) for t in list(unique_set_type_2_terms)]
        for unique_type_2_term in unique_list_type_2_terms:
            alternative_pairs.append(unique_type_2_term)

        unique_alternative_bigram_list = np.unique([j for i in alternative_pairs for j in i]).tolist()
        # non_alternative_bigram_not_in_all_parser = []
        non_alternative_bigram = []
        for bigram in merge_output:
            if bigram not in unique_alternative_bigram_list:
                non_alternative_bigram.append(bigram.replace('_', ' '))
                #####TEMPERORY COMMENTED, AS WE DONT'T NEED TO ADD SQUARE BETWEEN THE PAIR NOT OCCUR IN ALL PARSETS####

                # if bigram in stanza_mwt and bigram in spacy_mwt and bigram in corenlp_mwt:
                #     non_alternative_bigram.append(bigram.replace('_', ' '))
                # else:
                #     non_alternative_bigram_not_in_all_parser.append(bigram.replace('_', ' '))
        # if(len(non_alternative_bigram_not_in_all_parser) > 0):
        #     non_alternative_bigram.append(f'[{", ".join(non_alternative_bigram_not_in_all_parser)}]')
                
        non_alternative_pairs = ', '.join(non_alternative_bigram)

        if(len(alternative_pairs) > 0):   
            unique_list_alternatice_pairs = [', '.join(t) for t in alternative_pairs]
            unique_alternative_pair = [word.replace('_', ' ')for word in unique_list_alternatice_pairs]
            op_alternative_pairs = '\n'.join([f"{i+1}. {word}"for i, word in enumerate(unique_alternative_pair)])
        
        op_arr.append([
            mwt,                                                          # MWT
            non_alternative_pairs, # Approved Pairs
            non_alternative_pairs, # Non-alternative pairs
            op_alternative_pairs,                                         # Alternative Pairs
            ', '.join([word.replace('_', ' ') for word in spacy_mwt]),    # spaCy
            ', '.join([word.replace('_', ' ') for word in corenlp_mwt]),  # corenlp_mwt
            ', '.join([word.replace('_', ' ') for word in stanza_mwt])    # Stanza
        ])

    # create output dataframe and store in excel file
    op_df = pd.DataFrame(op_arr, columns=['MWT', 'Approved pairs', 'Non-alternative pairs', 'Alternative pairs', 'spaCy', 'CoreNLP', 'Stanza'])

    """
    Rearrange rows according to Vamshidhar code(sort by last two wotds of MWT)
    """
    # Remove duplicate data
    op_df = op_df.drop_duplicates()
    last_words = pd.Series(map(lambda x: " ".join(str(x).split()[-2:]), op_df['MWT']))
    unique_last_words = last_words.drop_duplicates()

    # creating a dictionary from the unique last words
    dic = dict.fromkeys(sorted(unique_last_words), [])

    for i in range(op_df.shape[0]):
        term = op_df.iloc[i,0]
        dic[" ".join(term.split()[-2:])] = dic[" ".join(term.split()[-2:])] + [i]

    # getting the index for reordering
    indices = sum(dic.values(),[])

    # reorder the op_df
    op_df = op_df.iloc[indices]
    op_df.reset_index(drop=True,inplace=True)

    # Save output file
    StyleFrame(op_df).to_excel(outputFile).close()