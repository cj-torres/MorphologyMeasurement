
from io import open
import pickle
import conllu
import os
import torch
import transformers

torch.cuda.empty_cache()
model = transformers.BertModel.from_pretrained("bert-base-multilingual-cased",output_hidden_states="true")
model.to("cuda")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")


class WordKey:

    def __init__(self, form, pos, token):
        self.form = form
        self.pos = pos
        self.token = token


def split_list(list, length):
    new_list = []
    split_num = -(-len(list)//length)-1
    for i in range(split_num):
        new_list.append(list[i*length:(i+1)*length])
    new_list.append(list[split_num*length:])

    return tuple(new_list)


def lang_preprocess(language,set):
    data_path = "ud-treebanks-v2.6/"
    S = set + ".conllu"
    L = "UD_" + language
    data = []
    pos_key = []
    for dir in os.listdir(data_path):
        if L in dir:
            print("Loading data " + data_path + dir)
            for file in os.listdir(data_path+dir):
                if file.endswith(S):
                    data_file = open(data_path + dir + "/" + file, "r", encoding="utf-8")
                    for j in conllu.parse_incr(data_file):
                        sentence = []
                        for i in j:
                            sentence.extend(tokenizer.tokenize(i["form"]))
                        data.append(torch.cuda.LongTensor(tokenizer.encode(sentence, pre_tokenized=True)))
                        pos_key.append([WordKey(i["form"],i["upos"],tokenizer.encode(i["form"])[1:-1]) for i in j])
    if not os.path.exists(("preprocessed/" + language)):
        os.mkdir("preprocessed/" + language)
    torch.save(torch.nn.utils.rnn.pad_sequence(data).transpose(0,1),"preprocessed/"+language+"/"+language+"_tensor.pt")
    pickle.dump(pos_key,open(("preprocessed/"+language+"/"+language+"_key.pk"),"wb"))
    del data
    del pos_key
    torch.cuda.empty_cache()


def sanity_check():
    data_file = open("ud-treebanks-v2.6/UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")

    for j in conllu.parse_incr(data_file):
        x = []
        for i in j:
            x.extend(tokenizer.encode(i["form"])[1:-1])
        print(len(tokenizer.encode([i["form"] for i in j])) == len(x) + 2)


def lang_process(language,pos):
    CHUNK_SIZE = 64
    key = split_list(pickle.load(open(("preprocessed/"+language+"/"+language+"_key.pk"),"rb")),CHUNK_SIZE)
    data = torch.load("preprocessed/"+language+"/"+language+"_tensor.pt").split(CHUNK_SIZE)
    for t in data:
        t.to("cpu")

    # Check that key and data are of matching size
    assert all([list(data[i].size())[0] == len(k) for i, k in enumerate(key)])

    vector_dictionary = {}
    with torch.no_grad():
        for chunk_index, chunk in enumerate(key):
            bert_output = model(data[chunk_index].to("cuda"))[2]
            print("Bert output loaded for chunk " + str(chunk_index) + ".")
            # Find and store list of word tokens to extract vectors for
            for sent_index, sent in enumerate(chunk):
                print("Sentence index: " + str(sent_index))
                word_index = 1
                for word in sent:
                    if word.pos == pos and len(word.token) == 1:
                        print(word.form + str(word.token))
                        if word.form.lower() not in vector_dictionary:
                            vector_dictionary[word.form.lower()] = []
                        vector_dictionary[word.form.lower()].append((bert_output[0][sent_index][word_index].to("cpu"),
                                                                               bert_output[8][sent_index][word_index].to("cpu"),
                                                                               bert_output[12][sent_index][word_index].to("cpu")))
                    word_index += len(word.token)
            data[chunk_index].to("cpu")
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
    if not os.path.exists(("postprocessed/" + language)):
        os.mkdir("postprocessed/" + language)
    pickle.dump(vector_dictionary, open(("postprocessed/" + language + "/" + language + "_" + pos + "_vecs.pk"), "wb"))


#def mean_and_deviance(language,pos):
 #   pickle.load(vector_dictionary, open(("postprocessed/" + language + "/" + language + "_" + pos + "_vecs.pk"), "wb"))
#
 #   for word in vector_dictionary:



# languages = ["Japanese","Korean","Russian"]
#
# for language in languages:
#     lang_preprocess(language,"train")
#     lang_process(language,"VERB")



#lang_preprocess("English","train")
#lang_process("English", "VERB")