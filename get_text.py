import requests
import urllib.parse
from selenium import webdriver
import os

# entity2id_file = './data/FB15K237/entity2id.txt'
# source_file = '../fb.txt'
# i2tu_file = './data/FB15K237/id2textrul.txt'
# i2t_file = './data/FB15K237/id2text.txt'
# pre_path = '<http://rdf.freebase.com/ns/'
#
# def get_id2texturl():
#     with open(entity2id_file, 'r') as f1, open(source_file, 'r') as f2, open(i2tu_file, 'w') as f3:
#         e2i = f1.readlines()
#         e2ul = f2.readlines()[4:]
#         e2u = {}
#         cnt = 0
#         for item in e2ul:
#             item = item.strip()[:-2].split('\t')
#             e2u[item[0]] = item[2]
#         for item in e2i[1:]:
#             item = item.split('\t')
#             fid = item[1]
#             item = item[0]
#             item = item.replace('/', '.')[1:]
#             item = pre_path + item + '>'
#             if item in e2u.keys():
#                 item = e2u[item]
#
#             else:
#                 cnt = cnt + 1
#                 item = ""
#
#             f3.write(fid.strip() + '\t' + item[1:-1] + '\n')
#         print(cnt)


# def url2text():
#     os.system('killall chromedriver')
#     chrome_options = webdriver.ChromeOptions()
#
#     chrome_options.add_argument('--proxy-server=socks5://localhost:1080')
#     driver = webdriver.Chrome('/home/doujh/bin/chromedriver', chrome_options=chrome_options)
#     driver.get('https://www.baidu.com')
#     print(driver.page_source)
    # with open(i2tu_file, 'r') as f1, open(i2t_file, 'w') as f2:
    #     i2tu = f1.readlines()
    #     for item in i2tu:
    #         item = item.split('\t')
    #         if len(item) == 1:
    #             f2.write(item)
    #         else:
    #             f2.write(item[0])
    #             driver.get('http://www.wikidata.org/entity/Q6718')
    #             print(driver.page_source)
    #             f2.write()


def get_desc():
    path1 = './Spider/download/'
    l = os.listdir(path1)
    temp_l = []
    for item in l:
        item.strip()
        if item[-4:] == 'json' or item[-4:] == 'html':
            temp_l.append(item)
    l = list(set(l) - set(temp_l))

    desc = {}
    for item in l:
        with open(path1 + item, 'r') as f:
            d = f.readlines()
            desc[item.split('_')[0]] = d[0]
    with open('./Spider/id2txt.txt', 'w') as fw:

        for i in range(14541):
            fw.write(str(i))
            if str(i) not in desc.keys():
                fw.write('\n')
            else:
                fw.write('\t')
                print(desc[str(i)])
                fw.write(desc[str(i)])
                fw.write('\n')



import random
def get_corpus():
    with open('./data/FB15K237/id2desc.txt', 'r') as f1, open('./data/FB15K237/sentence_rev.vec', 'r') as f2, open('./data/FB15K237/sentence.vec', 'w') as f3:
        a = f1.readlines()
        b = f2.readlines()[1:]
        pos = 0
        for item in a:
            item = item.split('\t')
            if len(item) == 1:
                f3.write(str(random.choice(b).split()[1:]))
            else:
                f3.write(str(b[pos].split()[1:]))
                pos = pos + 1
            f3.write('\n')

def get_corpus_wn():
    with open('./data/WN18/sentence.vec', 'r') as f1, open('./data/WN18/sent.vec', 'w') as f2:
        a = f1.readlines()
        a = a[1:]
        for item in a:
            f2.write(str(item.split()[1:]))
            f2.write('\n')

import torch
def get_pretrained_embed(path):
    res = []
    with open(os.path.join(path, 'sentence.vec'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            line = line[1:-1]
            line = line.replace("'", "").replace(" ", "").split(',')
            line = list(map(float, line))
            res.append(line)
    res = torch.tensor(res)
    return res


def wordnet_rr_corpus():
    with open('./data/WN18/entity2id.txt', 'r' ) as f1, open('./data/WN18RR/entity2id.txt', 'r') as f2, open('./data/WN18/sentence.vec', 'r') as f3, open('./data/WN18RR/sentence.vec', 'w') as f4:
        mp = {}
        mp2 = {}
        mp3 = {}
        lines = f3.readlines()
        for i in range(len(lines)):
            mp3[i] = lines[i]
        for line in f1.readlines():
            line = line.strip().split('\t')
            if len(line) == 1:
                continue
            elif len(line) == 2:
                mp[line[0]] = int(line[1])
        for line in f2.readlines():
            line = line.strip().split('\t')
            if len(line) == 1:
                continue
            elif len(line) == 2:
                if not line[0] in mp.keys():
                    print('not has key!')
                mp2[int(line[1])] = mp3[mp[line[0]]]
        for i in range(len(lines)):
            f4.write(mp2[i])







if __name__ == '__main__':
    # get_desc()
    # get_id2texturl()
    # url2text()
    # get_corpus_wn()
    # a = get_pretrained_embed('./data/FB15K237/')
    # print(a.shape)
    # get_corpus()
    wordnet_rr_corpus()

