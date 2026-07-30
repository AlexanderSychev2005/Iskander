import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertSelfAttention
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
from transformers.models.llama.configuration_llama import LlamaConfig

class BertSelfAttentionWithRoPE(BertSelfAttention):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        llama_config = LlamaConfig(
            hidden_size=self.attention_head_size * self.num_attention_heads,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.rotary_emb = LlamaRotaryEmbedding(
            config=llama_config,
        )

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        
        output_attentions = getattr(
            self.config, "output_attentions", output_attentions
        ) or kwargs.get("output_attentions", output_attentions)
        
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Apply RoPE
        seq_length = hidden_states.shape[1]
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=hidden_states.device
        )
        position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        cos, sin = self.rotary_emb(value_layer, position_ids)
        query_layer, key_layer = apply_rotary_pos_emb(
            query_layer, key_layer, cos, sin
        )
        query_layer = query_layer.to(value_layer.dtype)
        key_layer = key_layer.to(value_layer.dtype)

        # PyTorch 2.0 Scaled Dot-Product Attention (FlashAttention)
        # We need to reshape attention_mask to boolean or additive float mask compatible with SDPA.
        # HF attention_mask is already expanded to [batch, 1, seq_len, seq_len] with 0.0 and -10000.0
        # SDPA natively supports float additive masks
        
        # SDPA is known to have backward pass bugs on ROCm with Bfloat16 causing inf gradients.
        # Since our seq_len is small (256), manual attention is perfectly fast and 100% stable.
        
        # [batch, heads, seq, head_dim]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        
        dropout_p = self.dropout.p if self.training else 0.0
        if dropout_p > 0.0:
            attention_probs = nn.functional.dropout(attention_probs, p=dropout_p, training=self.training)
            
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, (attention_probs if output_attentions else None)


class AkkadianModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_hidden_layers=6, num_attention_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Text Encoder (NO Absolute Positional Embeddings)
        self.char_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.emb_norm = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(0.1)
        
        # 2. Text Transformer
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=2048, # Увеличен лимит для длинных табличек с RoPE
            is_decoder=False,
            position_embedding_type="none" # Отключаем абсолютные эмбеддинги BERT
        )
        self.encoder = BertEncoder(config)
        
        # Внедрение RoPE во все слои внимания
        for layer in self.encoder.layer:
            layer.attention.self = BertSelfAttentionWithRoPE(config)
        
        # 3. MLM Head
        self.restore_dense = nn.Linear(hidden_size, hidden_size)
        self.restore_act = nn.GELU()
        self.restore_norm = nn.LayerNorm(hidden_size)
        self.restore_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # 4. Unknown Gap Expansion Head ([#] token)
        self.unk_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2)
        )
        
        # 5. Metadata Classification Heads
        self.period_head = nn.Linear(hidden_size, 7)
        self.genre_head = nn.Linear(hidden_size, 6)
        self.language_head = nn.Linear(hidden_size, 4)
        self.provenience_head = nn.Linear(hidden_size, 7)
        
        # Apply strict BERT initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights properly for deep Transformers."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(self, input_ids, labels=None, unk_labels=None, period_labels=None, genre_labels=None, language_labels=None, provenience_labels=None, return_dict=True):
        # 1. Text Features (Чистые посимвольные эмбеддинги, без абсолютных позиций)
        x = self.char_embeddings(input_ids)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)
        
        # 2. Attention Mask
        pad_id = 0 # Assuming 0 is pad_id
        attn_mask = (input_ids != pad_id)
        
        extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2).to(dtype=x.dtype)
        extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0
        
        # 3. Transformer (RoPE вращает векторы прямо внутри self-attention)
        enc_out = self.encoder(
            x,
            attention_mask=extended_attn_mask,
            return_dict=True
        )
        seq = enc_out.last_hidden_state
        
        # 4. MLM Output
        x_res = self.restore_norm(self.restore_act(self.restore_dense(seq)))
        logits = (x_res @ self.char_embeddings.weight.T) + self.restore_bias
        
        # 5. Unknown Gap Expansion Output (per-token, but only trained on [#] positions)
        logits_unk = self.unk_head(seq)
        
        # 6. Metadata Outputs (using [CLS] token at index 0)
        cls_embed = seq[:, 0, :]
        period_logits = self.period_head(cls_embed)
        genre_logits = self.genre_head(cls_embed)
        language_logits = self.language_head(cls_embed)
        provenience_logits = self.provenience_head(cls_embed)
        
        # 7. Historical Context Embedding (emb_context)
        # Average of cls_embed and the mean of all sequence token embeddings
        seq_mean = seq[:, 1:, :].mean(dim=1)
        emb_context = (cls_embed + seq_mean) / 2.0
        
        loss = None
        if any(l is not None for l in [labels, unk_labels, period_labels, genre_labels, language_labels, provenience_labels]):
            loss_mlm_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)
            loss_unk_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_meta_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            loss = 0.0
            
            # MLM Loss (Weight = 3.0)
            if labels is not None:
                if (labels != -100).any():
                    loss += 3.0 * loss_mlm_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    
            # UNK Loss (Gap Expansion) (Weight = 1.0)
            if unk_labels is not None:
                if (unk_labels != -100).any():
                    loss += 1.0 * loss_unk_fct(logits_unk.view(-1, 2), unk_labels.view(-1))
            
            # Metadata Losses (Total Weight = 1.0 -> 0.25 each)
            meta_weight = 0.25
            if period_labels is not None and (period_labels != -100).any(): 
                loss += meta_weight * loss_meta_fct(period_logits, period_labels)
            if genre_labels is not None and (genre_labels != -100).any(): 
                loss += meta_weight * loss_meta_fct(genre_logits, genre_labels)
            if language_labels is not None and (language_labels != -100).any():
                loss += meta_weight * loss_meta_fct(language_logits, language_labels)
            if provenience_labels is not None and (provenience_labels != -100).any():
                loss += meta_weight * loss_meta_fct(provenience_logits, provenience_labels)
            
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "logits_unk": logits_unk,
                "emb_context": emb_context,
                "period_logits": period_logits,
                "genre_logits": genre_logits,
                "language_logits": language_logits,
                "provenience_logits": provenience_logits
            }
        
        return (loss, logits, logits_unk, emb_context, period_logits, genre_logits, language_logits, provenience_logits) if loss is not None else (logits, logits_unk, emb_context, period_logits, genre_logits, language_logits, provenience_logits)

if __name__ == "__main__":
    model = AkkadianModel(vocab_size=1529)
    print("Multi-task Model with RoPE and unk_head is valid!")
