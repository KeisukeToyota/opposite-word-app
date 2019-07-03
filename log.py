import MeCab
mecab= MeCab.Tagger("-Ochasen")
mecab.parseToNode('コアラのマーチ')
node = mecab.parseToNode('コアラのマーチ')
mecab.parse('コアラのマーチ')
print(mecab.parse('コアラのマーチ'))
print(mecab.parse('プラダを着た悪魔'))
node
node.feature
node
node.surface
node.surface
node = next(node)
node=  node.next
node.surface
node = node.next
node.surface
node.feature
node.feature.split(',')[0]
def parse_text(text):
    mecab = MeCab.Tagger("-Ochasen")
    node = mecab.parseToNode(text)

    change_list = []

    while node:
        if node.feature.split(',')[0] in ['名詞', '動詞']:
            change_list.append(opposite_word(node.surface))
        else:
            change_list.append(node.surface)
        node = node.next
    
    return ''.join(change_list)
parse_text('コアラのマーチ')
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
    result = model.most_similar(positive=[word, org], negative=[taigigo], topn=15)[0][0]
    return result
parse_text('コアラのマーチ')
parse_text('機動戦士ガンダム')
parse_text('赤の他人')
model = modelV
parse_text('赤の他人')
parse_text('機動戦士ガンダム')
parse_text('ゲスの極み乙女。')
model = modelW
parse_text('ゲスの極み乙女。')
model = modelV
parse_text('進撃の巨人')
parse_text('魔法少女')
parse_text('生理的に無理')
parse_text('ウサギは寂しいと死ぬ')
parse_text('やせ我慢')
parse_text('痩せ我慢')
parse_text('週刊少年ジャンプ')
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
check_hinsi('週刊少年ジャンプ')
def parse_text(text):
    
    change_list = []

    for t in check_hinsi(text):
        if t[1].split('-')[0] in ['名詞', '動詞']:
            change_list.append(opposite_word(node.surface))
        else:
            change_list.append(node.surface)
    
    return ''.join(change_list)
parse_text('週刊少年ジャンプ')
def parse_text(text):
    
    change_list = []

    for t in check_hinsi(text):
        if t[1].split('-')[0] in ['名詞', '動詞']:
            change_list.append(opposite_word(t[0]))
        else:
            change_list.append(t[0])
    
    return ''.join(change_list)
parse_text('週刊少年ジャンプ')
model = modelW
parse_text('週刊少年ジャンプ')
parse_text('冷やし中華始めました')
parse_text('死に物狂い')
parse_text('ちびまる子ちゃん')
parse_text('空を自由に飛びたいな、はい、タケコプター ')
parse_text('おまえがナンバー１だ！！')
parse_text('たったひとつの真実見抜く、見た目は子供、頭脳は大人、その名は名探偵コナン')
parse_text('そうだ、嬉しいんだ生きる喜び、たとえ胸の傷が痛んでも')
model = modelV
parse_text('そうだ、嬉しいんだ生きる喜び、たとえ胸の傷が痛んでも')
model = modelW
parse_text('石橋を叩いて渡る ')
parse_text('アウトオブ眼中')
parse_text('僕は新世界の神になる')
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
def most_niteiru(input_word, entries_list):
    similarities = [model.similarity(input_word, entry) for entry in entries_list]
    results = [[similarity, entry] for (entry, similarity) in zip(entries_list, similarities)]
    return sorted(results, reverse=True)
def opposite_word(word):
    df_alltaigigo_filterd = create_filterd_taigigo_list(word)
    df_alltaigigo_filterd = df_alltaigigo_filterd.sort_values(by='sim_inp_org', ascending=False)
    org = df_alltaigigo_filterd.iloc[0].org
    taigigo = df_alltaigigo_filterd.iloc[0].taigigo
    result = model.most_similar(positive=[word, org], negative=[taigigo], topn=15)
    return result
opposite_word('コアラ')
r = opposite_word('コアラ')
r
type(r)
r
list(map(lambda x: x[0], r))[:5]
get_natural_taigigo_NEO('コアラ', list(map(lambda x: x[0], opposite_word('コアラ')))[:5])
most_niteiru('コアラ', list(map(lambda x: x[0], opposite_word('コアラ')))[:5])
%history -f log.py
