from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math
import difflib

# Load model GPT2
model_id = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_id)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

def get_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

def similarity_score(original, new_text):
    return difflib.SequenceMatcher(None, original, new_text).ratio()

if __name__ == "__main__":
    ori = input("Masukkan teks asli (tekan Enter jika tidak ada): ")
    teks = input("Masukkan teks yang ingin dianalisis: ")

    perp = get_perplexity(teks)
    print(f"\n⚠️  Perplexity Score: {perp:.2f}")

    if ori.strip():
        sim = similarity_score(ori, teks)
        print(f"📎 Similarity ke teks asli: {sim*100:.2f}%")
        if 0.6 < sim < 0.95:
            print("➡️  Kemungkinan besar hasil parafrase otomatis.")
    else:
        if perp < 30:
            print("🤖 Diduga kuat teks ini dihasilkan AI.")
        else:
            print("🧠 Teks ini terlihat seperti buatan manusia.")
