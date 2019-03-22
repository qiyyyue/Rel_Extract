#Rel_Extract
    ##Bert_Utils:
        基于openAI bert model
        ###embedding：
            BertVector.py:
                BertVector.encode(sentences)
            example:
                '''
                from Bert_Utils.BertVector import BertVector
                bv = BertVector()
                bv.encode(['测试句子'])
                '''