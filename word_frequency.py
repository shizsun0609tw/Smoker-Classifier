import glob


def readFile(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        words = fileObj.read() #puts the file into an array
        fileObj.close()
        return words
output_file = 'out_data.txt'

f = open(output_file,'a')

rw_smoke = ['smoke','smoking','smoker', 'smokes', 'tobacco', 'cigarette']
rw_neg = ['quit', 'denies', 'deny', 'stop']
rw_2_words = ['denies tobacco', 'denies cigarette', 'not smoke', 'no smoking', 'any tobacco', 'no messes', 'non-smoker']
rw_temporal = ['former', 'prior', 'ago', 'past']
rw_disease = ['cough', 'coughing', 'chronic', 'pain']
rw_organ = ['chest', 'lungs']
rw_other = ['unknown',  'none']

represent_word = ['smoke','smoking','smoker', 'smokes', 'tobacco', 'cigarette', 'quit', 'no', 'not', 'denies', 'deny', 'unknown', 'non-smoker',
             'none','cough', 'coughing', 'chronic', 'chset', 'pain', 'lungs', 'stop']

represent_word_2 = ['denies tobacco', 'denies cigarette', 'not smoke', 'no smoking', 'any tobacco', 'no messes']

represent_word_3 = ['former', 'prior', 'ago', 'past']

weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1]

for file in glob.glob('*.txt'):
    w_idx = 0;
    if file == output_file:
        #print("yes  ")
        continue
    document = readFile(file)
    from itertools import chain
    #document = list(chain.from_iterable(document))
    #document = document.rave1()
    replace_set = {'\n', '/'}
    document = " ".join(" " if i in replace_set else i for i in document.split())
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    document = [document]
    X = vectorizer.fit_transform(document)
    freq_arr = X.toarray()
    feat_arr = vectorizer.get_feature_names()
    
    freq = {}
    for word in represent_word:
        freq[word] = 0
        if word in feat_arr:
            i = feat_arr.index(word)
            freq[word] = int(freq_arr[0][i])
        w_idx = w_idx + 1;
    #   f.write(file + '\n')    
    #f.write(str(freq) + '\n')

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    X2 = vectorizer.fit_transform(document)
    freq_arr_2 = X2.toarray()
    feat_arr_2 = vectorizer.get_feature_names()

    freq_2 = {}
    for words in represent_word_2:
        freq[words] = 0
        if words in feat_arr_2:
            i = feat_arr_2.index(words)
            freq[words] = int(freq_arr_2[0][i])
            
    if freq['smoke'] == 0 and freq['smoker'] == 0 and freq['smokes'] == 0 and freq['smoking'] == 0 and freq['tobacco'] == 0 and freq['cigarette'] == 0:
        freq['is_unknown'] = 1;
    else:
        freq['is_unknown'] = 0;
        
    for word in represent_word_3:
        freq[word] = 0
        if word in feat_arr:
            i = feat_arr.index(word)
            freq[word] = int(freq_arr[0][i])
        w_idx = w_idx + 1;
        
    #smoke    
    total_smoke = 0;    
    for word in rw_smoke:
        total_smoke = total_smoke + freq[word]
    
    #neg
    total_neg = 0;    
    for word in rw_neg:
        total_neg = total_neg + freq[word]
    #2_words 
    total_2words = 0;    
    for word in rw_2_words:
        total_2words = total_2words + freq[word]
    #temporal
    total_temp = 0;    
    for word in rw_temporal:
        total_temp = total_temp + freq[word]
    #disease
    total_disease = 0;    
    for word in rw_disease:
        total_disease = total_disease + freq[word]
    #organ    
    total_organ = 0;    
    for word in rw_organ:
        total_organ = total_organ + freq[word]
        
    total_other = 0;    
    for word in rw_other:
        total_other = total_other + freq[word]
    #f.write(str(total_freq_neg1) + '\n')    
    f.write(file + ',' + str(total_smoke) + ',' + str(total_neg) + ',' + str(total_2words) + ',' + str(total_temp) + ',' + str(total_disease) + ',' + str(total_organ) + ',' + str(total_other) + '\n')
    #f.write(str(freq) + '\n')



f.close()