#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: liyangmei
# date: 2019/4/23


import myutils


KEYWORDS = ["配偶", "丈夫", "现夫", "爱人", "结婚", "成婚", "老公", "结为", "妻", "夫人", "夫妻", "之夫", "之妻"
            "前夫", "离婚", "未婚夫", "订婚", "婚约", "定亲", "成亲", "结亲",
            "妻子", "现妻", "老婆", "前妻", "未婚妻", "太太",
            "血亲", "祖父", "爷爷", "爷孙", "姥爷", "外公", "外祖父",
            "祖母", "奶奶", "外祖父", "外祖母", "外婆", "姥姥",
            "父", "父亲", "爸爸", "爸", "妈", "生父", "母", "母亲", "妈妈", "生母", "之父", "之母",
            "父子", "爷俩", "娘俩", "母子", "母女", "父女",
            "儿子", "孩子", "女儿", "长子", "长女", "次子", "次女",
            "私生子", "私生女", "诞下", "之子", "之女", "亲生", "私生",
            "孙子", "孙女", "外孙", "外孙女", "之孙",
            "哥哥", "大哥", "哥", "兄长", "弟兄", "长兄", "表兄", "堂兄", "表哥", "堂哥", "表兄弟", "堂兄弟", "之兄",
            "姐姐", "大姐", "姐", "长姐", "姐妹", "表姐", "堂姐", "堂姐妹", "表姐妹", "之姐",
            "弟弟", "弟", "小弟", "表弟", "堂弟", "之弟",
            "妹妹", "妹", "小妹", "表妹", "堂妹", "之妹",
            "叔叔", "表叔", "堂叔", "小叔", "叔父", "叔祖",
            "伯伯", "大伯", "伯父", "叔伯", "伯祖",
            "舅舅", "表舅", "舅父", "外舅", "母舅",
            "姑姑", "大姑", "表姑", "姑妈", "姑母", "伯母",
            "侄子", "侄女", "外甥", "外甥女",
            "儿媳", "儿媳妇", "女婿", "姑爷", "嫂子", "大嫂", "嫂嫂",
            "公公", "岳父", "丈人", "老丈人",
            "朋友", "好朋友", "结义", "结义兄弟", "战友", "之友",
            "女朋友", "男朋友", "恋人", "恋爱", "爱慕", "挚爱", "至爱", "谈恋爱", "相恋", "相爱",
            "老师", "师父", "拜师", "师承",
            "学生", "徒弟", "师从"]
KEYWORDS = set(KEYWORDS)


def has_key_words(sentence):
    con = set(sentence) & KEYWORDS
    if len(con) != 0:
        return True


def same_family_name(name1, name2):
    if name1[0] == name2[0]:
        return True


def main(in_sent, in_label, out_file):
    sentence_list = myutils.rawdata2list(in_sent)  # [sentenceid,entity1,entity2,sentence]
    label_dict = myutils.rawdata2dict(in_label)  # key:句子编号id; value：关系int值

    result = []
    with open(out_file, "w", encoding="UTF-8") as fw:
        for s in sentence_list:
            label = label_dict[s[0]]
            pred = 0
            if has_key_words(s[3]) or same_family_name(s[1], s[2]):
                pred = 100
            result.append([s[0], label, pred])
            fw.write(s[0] + "\t" + str(label) + "\t" + str(pred) + "\t" + s[1] + "\t" + s[2] + "\n")

    result_0_0 = 0
    result_0_100 = 0
    result_100_0 = 0
    result_100_100 = 0
    for item in result:
        if item[1] == 0:
            if item[2] == 0:
                result_0_0 = result_0_0 + 1
            else:
                result_0_100 = result_0_100 + 1
        else:
            if item[2] == 0:
                result_100_0 = result_100_0 + 1
            else:
                result_100_100 = result_100_100 + 1
    print("result_0_0: %d" % result_0_0)
    print("result_0_100: %d" % result_0_100)
    print("result_100_0: %d" % result_100_0)
    print("result_100_100: %d" % result_100_100)


if __name__ == "__main__":
    print("begin...")
    # in_sent = "../../Data/RawData/sent_train.txt"
    # in_label = "../../Data/RawData/rawdata/sent_relation_train.txt"
    # out_file = "2all_train_result.txt"
    in_sent = "../../Data/RawData/rawdata/sent_dev.txt"
    in_label = "../../Data/RawData/rawdata/sent_relation_dev.txt"
    out_file = "2all_dev_result.txt"
    main(in_sent, in_label, out_file)
    print("done!")
