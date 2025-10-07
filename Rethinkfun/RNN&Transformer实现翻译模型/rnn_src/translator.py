import os
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#此部分代码为BPE模型训练，仅需使用一次
#spm.SentencePieceTrainer.Train('--input="..\\data\\train_en.txt" --model_prefix=en_bpe --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
#spm.SentencePieceTrainer.Train('--input="..\\data\\train_zh.txt" --model_prefix=zh_bpe --vocab_size=16000 --model_type=bpe --character_coverage=0.9995 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
#加载SentencePiece模型
sp_en = spm.SentencePieceProcessor()
sp_en.load('en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('zh_bpe.model')
#定义取词器
def tokenize_en(text):
    return sp_en.encode(text, out_type=int)

def tokenize_cn(text):
    return sp_cn.encode(text, out_type=int)

#提取特殊token的ID
PAD_ID = sp_en.pad_id()
UNK_ID = sp_en.unk_id()
BOS_ID = sp_en.bos_id()
EOS_ID = sp_en.eos_id()

#定义数据集Dataset与加载器DataLoader
class TranslationDataset(Dataset):
    #定义数据集参数
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer, max_len=100):
        with open(src_file, 'r', encoding='utf-8') as f:
            src_lines = f.read().splitlines()
        with open(trg_file, 'r', encoding='utf-8') as f:
            trg_lines = f.read().splitlines()
        assert len(src_lines) == len(trg_lines)
        self.pairs = []
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        #在每个句子前后增加<BOS>,<EOS>
        for src, trg in zip(src_lines, trg_lines):
            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]
            #只保留小于max_len的输入输出序列
            if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                self.pairs.append((src_ids, trg_ids))
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)

    #collate the tokens:add some <PAD>, and return the tensors
    @staticmethod #
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens


dataset = TranslationDataset('..\\data\\train_en.txt', '..\\data\\train_zh.txt', tokenize_en, tokenize_cn)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=TranslationDataset.collate_fn)
for src, trg, _, _ in loader:
    print(src.shape, trg.shape)
    print(src, trg)
    break


