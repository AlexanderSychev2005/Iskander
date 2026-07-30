import json
import os
from collections import Counter

class CharacterTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.mask_token = '[MASK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.hash_token = '[#]'
        self.x_token = 'x'
        
        self.special_tokens = [
            self.pad_token,
            self.unk_token,
            self.mask_token,
            self.cls_token,
            self.sep_token,
            self.hash_token,
            self.x_token
        ]
        
        # Initialize vocab with special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token

    def build_vocab(self, jsonl_path, min_freq=1):
        """Builds vocabulary from a JSONL dataset file."""
        print(f"Building vocabulary from {jsonl_path}...")
        char_counter = Counter()
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    signs = data.get('signs', '')
                    if signs:
                        char_counter.update(signs)
                except Exception:
                    pass
                    
        # Sort characters by frequency
        valid_chars = [char for char, count in char_counter.most_common() if count >= min_freq]
        print(f"Found {len(valid_chars)} unique characters with frequency >= {min_freq}")
        
        # Add to vocab
        start_idx = len(self.special_tokens)
        for i, char in enumerate(valid_chars):
            idx = start_idx + i
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
            
    def save(self, vocab_path):
        """Saves vocabulary to a JSON file."""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {vocab_path} (Size: {len(self.vocab)})")
        
    def save_pretrained(self, save_directory):
        """Hugging Face Trainer compatibility."""
        os.makedirs(save_directory, exist_ok=True)
        self.save(os.path.join(save_directory, "vocab.json"))
        
    def load(self, vocab_path):
        """Loads vocabulary from a JSON file."""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        print(f"Vocabulary loaded from {vocab_path} (Size: {len(self.vocab)})")
        
    def encode(self, text, add_special_tokens=True, max_length=None):
        """Encodes a string into a list of token IDs."""
        if text is None:
            text = ""
            
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.cls_token])
            
        for char in text:
            token_ids.append(self.vocab.get(char, self.vocab[self.unk_token]))
            
        if add_special_tokens:
            token_ids.append(self.vocab[self.sep_token])
            
        if max_length is not None:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                # Ensure SEP token is still at the end if truncated
                if add_special_tokens:
                    token_ids[-1] = self.vocab[self.sep_token]
            else:
                # Pad
                pad_len = max_length - len(token_ids)
                token_ids.extend([self.vocab[self.pad_token]] * pad_len)
                
        return token_ids
        
    def decode(self, token_ids, skip_special_tokens=False):
        """Decodes a list of token IDs back into a string."""
        chars = []
        for idx in token_ids:
            char = self.inverse_vocab.get(idx, self.unk_token)
            if skip_special_tokens and char in self.special_tokens:
                continue
            chars.append(char)
        return "".join(chars)

# Helper for testing/building
if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    dataset_path = r"C:\Programming\akkadian\data\oracc_dataset\train.jsonl"
    vocab_path = r"C:\Programming\akkadian\data\oracc_dataset\vocab.json"
    
    tokenizer.build_vocab(dataset_path)
    tokenizer.save(vocab_path)
    
    # Test encoding
    test_text = "𒀀𒈾𒈗x" # Includes 'x' which is our gap token!
    encoded = tokenizer.encode(test_text, max_length=10)
    print(f"Test Text: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded, skip_special_tokens=False)}")
