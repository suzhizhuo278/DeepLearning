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

"""
dataset = TranslationDataset('..\\data\\train_en.txt', '..\\data\\train_zh.txt', tokenize_en, tokenize_cn)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=TranslationDataset.collate_fn)
for src, trg, _, _ in loader:
    print(src.shape, trg.shape)
    print(src, trg)
    break
"""
#---------------复制------------------------------------------------

# ---------------------#
# 3. Model Definitions with Attention
# ---------------------#
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        #第一层输入维度为Encoder的输出隐状态（因为是双向的，所以维度为hid_dim*2,）和Decoder的输入隐状态（单向，维度为hid_dim—）的拼接。
        self.attn = nn.Linear(hid_dim * 2 + hid_dim, hid_dim)
        #输出一个代表注意力的logit值。
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # 调整Decoder当前时间步输入隐状态的维度: [1, batch, hid_dim] -> [batch, 1, hid_dim]
        hidden = hidden.permute(1, 0, 2)
        # 调整encoder各个时间步输出隐状态的维度： [src_len, batch, hid_dim*2] -> [batch, src_len, hid_dim*2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        src_len = encoder_outputs.shape[1]
        # 中文当前一个token需要和英文所有token计算注意力。所以需要把中文token的状态复制多份，以便进行统一拼接。
        # 因为Decoder只有当前时间步输入的隐状态，复制到和Encoder输出隐状态同样的src_len。
        hidden = hidden.repeat(1, src_len, 1)  # [batch, src_len, hid_dim]

        # 拼接Decoder当前输入的隐状态和Encoder在各个时间步输出的隐状态，然后经过一个线性层，tanh激活。
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hid_dim]
        # 输出Decoder当前中文token与所有英文token的注意力值。
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        # mask标志哪些位置为<pad>,对于填充的位置，注意力值为一个大的负值。这样经过softmax就为0。
        attention = attention.masked_fill(mask == 0, -1e10)
        # 利用softmax将注意力的值归一化。让生成当前中文token对输入的英文各个token的注意力之和为1。
        return torch.softmax(attention, dim=1)  # [batch, src_len]


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=3):
        super().__init__()
        # 默认3个循环层
        self.n_layers=n_layers
        # 定义Embedding,可以将BPE分词输出的token id，转化为emd_dim的embedding向量。<pad>不参与运算，它的embedding不需要学习。
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        # 定义双向LSTM模型
        self.bi_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True)
        # 定义线性层列表来降低维度。因为每个Encoder是双向的，隐状态和细胞状态为hid_dim*2，Decoder是单向的，隐状态和细胞状态维度为hid_dim。
        self.fc_hidden = nn.ModuleList([nn.Linear(hid_dim * 2, hid_dim) for _ in range(n_layers)])
        self.fc_cell = nn.ModuleList([nn.Linear(hid_dim * 2, hid_dim) for _ in range(n_layers)])

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        #将一个 padded sequence（已经填充到统一长度的 batch 序列） 转换为一个特殊的 PackedSequence 对象
        #这个对象在传入 RNN 时能跳过 padding 部分的计算。
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        #outputs,形状为 (seq_len, batch_size, hid_dim*2),表示每个时间步、最后一层LSTM的双向隐状态拼接。
        #(hidden, cell) ，形状都为(num_layers * 2, batch_size, hid_dim)表示每一层、每个方向在最后一个时间步的隐状态或细胞状态。
        outputs, (hidden, cell) = self.bi_lstm(packed)

        #将 PackedSequence 类型的输出还原成带 padding 的标准 Tensor，方便后续处理。
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)  # [src_len, batch, hid_dim*2]

        # 重塑隐藏状态和细胞状态: [n_layers * 2, batch, hid_dim] -> [n_layers, 2, batch, hid_dim]
        hidden = hidden.view(self.n_layers, 2, -1, hidden.size(2))
        cell = cell.view(self.n_layers, 2, -1, cell.size(2))

        # 为每一层处理前向和后向状态
        final_hidden = []
        final_cell = []

        for layer in range(self.n_layers):
            # 对LSTM每一层最后一个时间步，将隐状态的正向和反向状态合并，细胞状态的正向和反向状态合并，分别通过一个线性层将维度从hid_dim*2降低为hid_dim维度。
            h_cat = torch.cat((hidden[layer][-2], hidden[layer][-1]), dim=1)
            c_cat = torch.cat((cell[layer][-2], cell[layer][-1]), dim=1)
            h_layer = torch.tanh(self.fc_hidden[layer](h_cat)).unsqueeze(0)
            c_layer = torch.tanh(self.fc_cell[layer](c_cat)).unsqueeze(0)

            final_hidden.append(h_layer)
            final_cell.append(c_layer)
        # 调整好维度为hid_dim的隐状态和细胞状态，可以传递给Decoder。
        hidden_concat = torch.cat(final_hidden, dim=0)
        cell_concat = torch.cat(final_cell, dim=0)
        return outputs, hidden_concat, cell_concat


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, n_layers=3):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_ID)
        # 单向LSTM，输入维度为注意力加权后的，Encoder输出隐状态维度（hid_dim*2）加上输入token的embedding的维度emb_dim。
        self.rnn = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim,num_layers=n_layers)
        # 定义最终分类头，输入为3倍的hid_dim，输出为字典大小。
        # 为什么输入是3倍的hid_dim呢？
        # 因为预测中文token时，输入不光是Decoder的最后一层的隐状态，还拼接了注意力向量。
        # 这样对这个时间步生成翻译的中文token会有帮助。而注意力向量来自双向的Encoder，所以要额外加上一个2倍的hid_dim.
        self.fc_out = nn.Linear(hid_dim*3, output_dim)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        # input_token: [batch]
        # 输入单个字符，增加一个维度
        input_token = input_token.unsqueeze(0)  # [1, batch]
        # 获取token的embedding
        embedded = self.embedding(input_token)  # [1, batch, emb_dim]
        # 获取当前步的输入隐状态
        last_hidden = hidden[-1].unsqueeze(0)
        # 当前步对所有encoder输出的注意力
        a = self.attention(last_hidden, encoder_outputs, mask)  # [batch, src_len]
        a = a.unsqueeze(1)  # [batch, 1, src_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)  # [batch, src_len, enc_hid_dim*2]

        # 用矩阵乘法获取注意力上下文向量
        weighted = torch.bmm(a, encoder_outputs)  # [batch, 1, enc_hid_dim*2]
        weighted = weighted.permute(1, 0, 2)  # [1, batch, enc_hid_dim*2]
        # 拼接输入token编码向量和注意力上下文向量
        lstm_input = torch.cat((embedded, weighted), dim=2)  # [1, batch, emb_dim + enc_hid_dim*2]
        # 一次只执行lstm的一个时间步。
        output, (hidden, cell) = self.rnn(lstm_input, (hidden, cell))  # output: [1, batch, hid_dim]

        # 移除第0维（第一个维度）
        output = output.squeeze(0)  # [batch, hid_dim]
        weighted = weighted.squeeze(0)
        # 拼接输出和注意力上下文向量，进入分类头，计算分类logtis
        prediction = self.fc_out(torch.cat((output, weighted), dim=1))  # [batch, output_dim]

        return prediction, hidden, cell, a.squeeze(1)  # attention weights for visualization



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        # 调用encoder
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        input_token = trg[0]
        mask = (src != PAD_ID).permute(1, 0)
        # 逐步调用Decoder
        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[t] = output
            input_token = trg[t]

        return outputs


# ---------------------#
# 4. Training
# ---------------------#
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    step_loss = 0  # 用于累计每个step的loss
    step_count = 0  # 当前step计数器

    for i, (src, trg, src_len, _) in enumerate(iterator):
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        # 更新损失统计
        step_loss += loss.item()
        epoch_loss += loss.item()
        step_count += 1

        # 每100个step打印一次
        if (i + 1) % 100 == 0:
            avg_step_loss = step_loss / step_count
            print(f'Step [{i + 1}/{len(iterator)}] | Loss: {avg_step_loss:.4f}')
            step_loss = 0  # 重置step损失
            step_count = 0  # 重置step计数器

    return epoch_loss / len(iterator)  # 返回整个epoch的平均loss


# ---------------------#
# 5. Main
# ---------------------#
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = TranslationDataset('../data/train_en.txt', '../data/train_zh.txt', tokenize_en, tokenize_cn)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=TranslationDataset.collate_fn)

    INPUT_DIM = sp_en.get_piece_size()
    OUTPUT_DIM = sp_cn.get_piece_size()
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 512

    attention = Attention(HID_DIM).to(device)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM).to(device)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load('seq2seq_bpe_attention.pt', map_location=device))
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    N_EPOCHS = 1

    for epoch in range(N_EPOCHS):
        loss = train(model, loader, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss:.4f}')
        torch.save(model.state_dict(), 'seq2seq_bpe_attention.pt')