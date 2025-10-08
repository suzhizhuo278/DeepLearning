import torch
import sentencepiece as spm
from transformer import build_transformer  # adjust import path as needed

# ---------------------#
# 1. Load Tokenizers
# ---------------------#
sp_en = spm.SentencePieceProcessor()
sp_en.load('en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('zh_bpe.model')

PAD_ID = sp_en.pad_id()  # 1
UNK_ID = sp_en.unk_id()  # 0
BOS_ID = sp_en.bos_id()  # 2
EOS_ID = sp_en.eos_id()  # 3

# ---------------------#
# 2. Load Trained Model
# ---------------------#
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model hyperparameters (must match training)
SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000
SRC_SEQ_LEN = 128
TGT_SEQ_LEN = 128

model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load('transformer.pt', map_location=DEVICE))  # load your trained model
model.eval()


# ---------------------#
# 3. Translation Function (Greedy)
# ---------------------#
def create_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (1, 1, 1, src_len)


def translate_sentence(sentence, max_len=100):
    # Tokenize and convert to IDs
    tokens = [BOS_ID] + sp_en.encode(sentence, out_type=int) + [EOS_ID]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)  # [1, src_len]
    src_mask = create_mask(src_tensor, PAD_ID)

    # Initialize target with <bos> token
    trg_indices = [BOS_ID]

    with torch.no_grad():
        # Encode the source sentence
        encoder_output = model.encode(src_tensor, src_mask)

        # Generate translation token by token
        for _ in range(max_len):
            trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(DEVICE)  # [1, current_trg_len]

            # Create target mask
            trg_mask = torch.tril(torch.ones((len(trg_indices), len(trg_indices)), device=DEVICE)).bool()
            trg_mask = trg_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, trg_len, trg_len]

            # Decode
            decoder_output = model.decode(encoder_output, src_mask, trg_tensor, trg_mask)
            output = model.project(decoder_output)

            # Get the last predicted token
            pred_token = output.argmax(2)[:, -1].item()
            trg_indices.append(pred_token)

            if pred_token == EOS_ID:
                break

    # Convert token IDs to text (skip <bos> and <eos>)
    translated = sp_cn.decode(trg_indices[1:-1])
    return translated


# ---------------------#
# 4. Interactive Test
# ---------------------#
if __name__ == '__main__':
    print("Transformer Translator (type 'quit' or 'exit' to end)")
    while True:
        src_sent = input("\nEnter English sentence: ")
        if src_sent.lower() in ['quit', 'exit']:
            break
        translation = translate_sentence(src_sent)
        print(f"Chinese Translation: {translation}")