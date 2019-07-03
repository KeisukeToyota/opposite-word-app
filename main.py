from gensim.models.word2vec import Word2Vec
import gensim
import pandas as pd
import MeCab

modelW_path = './models/word2vec.gensim.model'
modelW = Word2Vec.load(modelW_path)

modelV_path = './models/model.vec'
modelV = gensim.models.KeyedVectors.load_word2vec_format(modelV_path, binary=False)

model =  modelW

df_alltaigigo = pd.read_csv('taigigo.csv')

def create_filterd_taigigo_list(input_word):
    df_alltaigigo_filterd = pd.DataFrame(index=[], columns = ["org","taigigo","sim_inp_org","sim_inp_tai","sim_org_tai"])
    for index, rows in df_alltaigigo.iterrows():
        try:
            sim_INP_ORG = model.similarity(input_word, rows[0])
            sim_INP_TAI = model.similarity(input_word, rows[1])
            sim_ORG_TAI = model.similarity(rows[0], rows[1])            
        except KeyError:
            continue

        series = pd.Series([rows[0], rows[1], sim_INP_ORG, sim_INP_TAI, sim_ORG_TAI], index = df_alltaigigo_filterd.columns )

        df_alltaigigo_filterd = df_alltaigigo_filterd.append(series, ignore_index = True)

    return df_alltaigigo_filterd

def opposite_word(word):
    df_alltaigigo_filterd = create_filterd_taigigo_list(word)
    df_alltaigigo_filterd = df_alltaigigo_filterd.sort_values(by='sim_inp_org', ascending=False)
    org = df_alltaigigo_filterd.iloc[0].org
    taigigo = df_alltaigigo_filterd.iloc[0].taigigo
    result = model.most_similar(positive=[word, org], negative=[taigigo], topn=15)
    return result

def check_hinsi(taisyou_word):
    mecab = MeCab.Tagger("-Ochasen")
    WordHinsi = []
    trline = taisyou_word.replace(u'\t', u'　')
    parsed_line = mecab.parse(trline)
    wordsinfo_list = parsed_line.split('\n')

    for wordsinfo in wordsinfo_list:
        info_list = wordsinfo.split('\t')
        if(len(info_list)>2):
            WordHinsi.append((info_list[0], info_list[3]))

    return WordHinsi

def most_niteiru(input_word, entries_list):
    similarities = [model.similarity(input_word, entry) for entry in entries_list]
    results = [[similarity, entry] for (entry, similarity) in zip(entries_list, similarities)]
    return sorted(results, reverse=True)

def parse_text(text):
    
    change_list = []

    for t in check_hinsi(text):
        if t[1].split('-')[0] in ['名詞', '動詞']:
            change_list.append(most_niteiru(t[0], list(map(lambda x: x[0], opposite_word(t[0])))[:5])[0][1])
        else:
            change_list.append(t[0])
    
    return ''.join(change_list)

# df_alltaigigo_filterd = df_alltaigigo.sort_values(by='sort_val', ascending=False)


def get_natural_taigigo_NEO(input_word, input_hinsi):
    df_alltaigigo_filterd = create_filterd_taigigo_list(input_word)
    FinalKouhoList = []

    for index, rows in df_alltaigigo_filterd.iterrows():
        enzan_kekka_list = model.most_similar(positive = [input_word, rows['taigigo']], negative=[rows['org']], topn=15)
        KouhoList = []

        for enzan_kekka in enzan_kekka_list:
            enzan_kekka_hinsi_list = check_hinsi(enzan_kekka[0])
            if( len(enzan_kekka_hinsi_list) == 1 ):
                if(input_hinsi[:2] == enzan_kekka_hinsi_list[0][1][:2]):
                    KouhoList.append(enzan_kekka_hinsi_list[0][0])

        if( len(KouhoList)>1 ):
            FinalKouhoList.append( KouhoList[0] )
            FinalKouhoList.append( KouhoList[1] )

        if(len(FinalKouhoList) > 10):
            break
    
    if(len(FinalKouhoList)<1):
        FinalKouhoList.append( input_word )

    FinalKouhoList_unique = list(set(FinalKouhoList))
    nita_word_list = most_niteiru(input_word, FinalKouhoList_unique)[:3]

    return nita_word_list

