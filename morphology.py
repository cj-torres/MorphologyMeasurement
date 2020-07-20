from io import open
import pickle
import conllu
import os
import torch
import transformers
from statistics import mean

torch.cuda.empty_cache()
model = transformers.BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states="true")
model.to("cuda")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-cased")


class WordKey:

    def __init__(self, form, pos, token, lemma):
        self.form = form
        self.pos = pos
        self.token = token
        self.lemma = lemma


def split_list(list, length):
    new_list = []
    split_num = -(-len(list) // length) - 1
    for i in range(split_num):
        new_list.append(list[i * length:(i + 1) * length])
    new_list.append(list[split_num * length:])

    return tuple(new_list)


def lang_preprocess(language, set):
    data_path = "ud-treebanks-v2.6/"
    S = set + ".conllu"
    L = "UD_" + language
    data = []
    pos_key = []
    for dir in os.listdir(data_path):
        if L in dir:
            print("Loading data " + data_path + dir)
            for file in os.listdir(data_path + dir):
                if file.endswith(S):
                    data_file = open(data_path + dir + "/" + file, "r", encoding="utf-8")
                    for j in conllu.parse_incr(data_file):
                        sentence = []
                        for i in j:
                            sentence.extend(tokenizer.tokenize(i["form"]))
                        data.append(torch.cuda.LongTensor(tokenizer.encode(sentence, pre_tokenized=True)))
                        pos_key.append(
                            [WordKey(i["form"], i["upos"], tokenizer.encode(i["form"])[1:-1], i["lemma"]) for i in j])
    if not os.path.exists(("preprocessed/" + language)):
        os.mkdir("preprocessed/" + language)
    torch.save(torch.nn.utils.rnn.pad_sequence(data).transpose(0, 1),
               "preprocessed/" + language + "/" + language + "_tensor.pt")
    pickle.dump(pos_key, open(("preprocessed/" + language + "/" + language + "_key.pk"), "wb"))
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


def lang_process(language, pos):
    CHUNK_SIZE = 32
    key = split_list(pickle.load(open(("preprocessed/" + language + "/" + language + "_key.pk"), "rb")), CHUNK_SIZE)
    data = torch.load("preprocessed/" + language + "/" + language + "_tensor.pt").split(CHUNK_SIZE)
    for t in data:
        t.to("cpu")

    # Check that key and data are of matching size
    assert all([list(data[i].size())[0] == len(k) for i, k in enumerate(key)])
    vector_dictionary = {}

    # word_vector_dictionary = {}
    # lemma_vector_dictionary = {}
    with torch.no_grad():
        for chunk_index, chunk in enumerate(key):
            bert_output = model(data[chunk_index].to("cuda"))[2]
            print("Bert output loaded for chunk " + str(chunk_index) + ".")
            # Find and store list of word tokens to extract vectors for
            for sent_index, sent in enumerate(chunk):
                word_index = 1
                for word in sent:
                    if word.pos == pos and len(word.token) == 1:

                        if word.lemma.lower() not in vector_dictionary:
                            vector_dictionary[word.lemma.lower()] = {}
                            vector_dictionary[word.lemma.lower()][word.form.lower()] = []

                        elif word.form.lower() not in vector_dictionary[word.lemma.lower()]:
                            vector_dictionary[word.lemma.lower()][word.form.lower()] = []

                        vector_dictionary[word.lemma.lower()][word.form.lower()].append(
                            (bert_output[0][sent_index][word_index].to("cpu"),
                             bert_output[8][sent_index][word_index].to("cpu"),
                             bert_output[12][sent_index][word_index].to("cpu")))

                        # if word.form.lower() not in word_vector_dictionary:
                        #     word_vector_dictionary[word.form.lower()] = []
                        # word_vector_dictionary[word.form.lower()].append((bert_output[0][sent_index][word_index].to("cpu"),
                        #                                                        bert_output[8][sent_index][word_index].to("cpu"),
                        #                                                        bert_output[12][sent_index][word_index].to("cpu")))
                        #
                        # if word.lemma.lower() not in lemma_vector_dictionary:
                        #     lemma_vector_dictionary[word.lemma.lower()] = []
                        # lemma_vector_dictionary[word.lemma.lower()].append((bert_output[0][sent_index][word_index].to("cpu"),
                        #                                                        bert_output[8][sent_index][word_index].to("cpu"),
                        #                                                        bert_output[12][sent_index][word_index].to("cpu")))

                    word_index += len(word.token)
            data[chunk_index].to("cpu")
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
    if not os.path.exists(("postprocessed/" + language)):
        os.mkdir("postprocessed/" + language)
    pickle.dump(vector_dictionary, open(("postprocessed/" + language + "/" + language + "_" + pos + "_vecs.pk"), "wb"))

    # pickle.dump(lemma_vector_dictionary, open(("postprocessed/" + language + "/" + language + "_" + pos + "_lemma_vecs.pk"), "wb"))


def vec_mean_dev(language, pos):
    dict = pickle.load(open(("postprocessed/" + language + "/" + language + "_" + pos + "_vecs.pk"), "rb"))
    # lemma_dict = pickle.load(open(("postprocessed/"+language+"/"+language+"_"+pos+"_lemma_vecs.pk"), "rb"))

    print("Language " + language + " loaded.")

    mean_dev_dict = {"LEM": {}, "WRD": {}}

    for lemma in dict:
        all_lemma = []
        for word in dict[lemma]:
            all_lemma.extend(dict[lemma][word])
            mean_dev_dict["WRD"][word + "_MEAN_DEV_WRD"] = word_mean_deviance(dict[lemma][word])

        mean_dev_dict["LEM"][lemma + "_MEAN_DEV_LEM"] = word_mean_deviance(all_lemma)

    pickle.dump(mean_dev_dict,
                open(("postprocessed/" + language + "/" + language + "_" + pos + "_MDDict.pk"), "wb"))
    # pickle.dump(dict_mean_deviance(word_dict),open(("postprocessed/"+language+"/"+language+"_"+pos+"_word_MD.pk"), "wb"))
    # pickle.dump(dict_mean_deviance(lemma_dict),open(("postprocessed/"+language+"/"+language+"_"+pos+"_lemma_MD.pk"), "wb"))


def word_mean_deviance(word):
    # mean_dev_dict = {}
    # for k in dictionary:
    # print("Processing " + word)
    mean_list = []
    dev_list = []
    for i, _ in enumerate(word[0]):
        vec_slice = [vecs[i] for vecs in word]
        mean_vec = torch.mean(torch.stack(vec_slice), 0)
        dist = []
        for vec in vec_slice:
            dist.append(torch.dist(mean_vec, vec, 2).item())
        deviance = mean(dist)
        mean_list.append(mean_vec)
        dev_list.append(deviance)
    assert len(mean_list) == 3
    n = len(word)

    return (n, torch.stack(mean_list), torch.FloatTensor(dev_list))
    # return mean_dev_dict


def average_deviance(dictionary, min_n):
    return torch.mean(torch.stack([dictionary[k][2] for k in dictionary if dictionary[k][0] >= min_n]), 0)


def comparison(language1, language2, pos, min_n=25, lemma=False):
    language1_MD = pickle.load(open(("postprocessed/" + language1 + "/" + language1 + "_" + pos + "_MDDict.pk"), "rb"))
    language2_MD = pickle.load(open(("postprocessed/" + language2 + "/" + language2 + "_" + pos + "_MDDict.pk"), "rb"))

    # else:
    #     language1_MD = pickle.load(
    #         open(("postprocessed/"+language1+"/"+language1 + "_" + pos + "_word_MD.pk"), "rb"))
    #     language2_MD = pickle.load(
    #         open(("postprocessed/"+language2+"/"+language2 + "_" + pos + "_word_MD.pk"), "rb"))
    if lemma:
        print(average_deviance(language1_MD["LEM"], min_n))
        print(average_deviance(language2_MD["LEM"], min_n))
    else:
        print(average_deviance(language1_MD["WRD"], min_n))
        print(average_deviance(language2_MD["WRD"], min_n))

# languages = ["Vietnamese","Turkish"]
#
# for language in languages:
#     lang_preprocess(language,"train")
#     lang_process(language,"VERB")
#     lang_process(language,"NOUN")
#     vec_mean_dev(language, "VERB")
#     vec_mean_dev(language,"NOUN")
#
# comparison("Russian","English","NOUN")
# comparison("Russian","Chinese","NOUN")
# comparison("Chinese","Italian","VERB")
# comparison("Chinese","German","VERB")

# lang_preprocess("English","train")
# lang_process("English", "VERB")
