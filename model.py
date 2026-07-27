"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    d={}
    for i,e in enumerate(specials):
        d[e]=i
    c=4
    for sent in sentences:
        for word in sent.split(" "):
            if word not in d:
                d[word]=c
                c+=1
    return d

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    d={}
    for e in token_to_id:
        d[token_to_id[e]]=e
    return d

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    ids=[token_to_id.get(e,token_to_id[unk_token]) for e in sentence.split(" ")]
    return ids if len(sentence) else []

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    res=[id_to_token[e] for e in ids]
    return res
    pass

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    n=min(len(ids),max_len)
    ids=ids[:n]
    for _ in range(max_len-n):
        ids.append(pad_id)
    return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    batch=torch.as_tensor(padded_sequences,dtype=torch.long)
    return batch

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    embeddings=torch.tensor(embeddings)
    d_model=torch.tensor(d_model)
    scale=torch.sqrt(d_model)
    embeddings*=scale
    return embeddings

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    pos=torch.arange(start=0,end=d_model,step=2,dtype=torch.long)
    const=-torch.log(torch.tensor(10000))/d_model
    freq=torch.exp(const * pos)
    return freq

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    res=torch.arange(start=0,end=max_len,step=1,dtype=torch.float32)
    res=torch.reshape(res,(max_len,1))
    res=torch.tensor(res,dtype=torch.float32)
    return res

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    seq_len,d_model=pe.size()
    pe[:,0:d_model:2]=torch.sin(position*div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    seq_len,d_model=pe.size()
    pe[:,1:d_model:2]=torch.cos(position*div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe=torch.zeros((max_len,d_model),dtype=torch.float32)
    div_term=compute_positional_div_term(d_model)
    position=build_position_index_column(max_len)
    pe=fill_even_indices_with_sin(pe, position, div_term)
    pe=fill_odd_indices_with_cos(pe, position, div_term)
    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    b,seq_len,d_model=embedded_batch.size()
    res=embedded_batch[:,:,:]+positional_encoding[:seq_len,:]
    return res

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    B,L=token_ids.size()
    pad_mask=token_ids != pad_id
    res=torch.reshape(pad_mask,(B,1,1,L))
    return res

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    causal_mask=torch.ones((1,1,seq_len,seq_len),dtype=torch.bool)
    res=torch.tril(causal_mask)
    return res

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    B,_,_,L=padding_mask.size()
    res=torch.zeros((B,1,L,L),dtype=torch.bool)
    res=padding_mask & causal_mask
    return res

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    att_score=torch.matmul(query,torch.transpose(key,-2,-1))
    return att_score

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    scores/=math.sqrt(d_k)
    return scores

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    res=torch.where(mask==True,scores,float("-inf"))
    return res

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    fully_masked=torch.isneginf(masked_scores).all(dim=-1,keepdim=True)
    masked_scores=masked_scores.masked_fill(fully_masked,0.0)
    res=torch.nn.functional.softmax(masked_scores,dim=-1,dtype=torch.float32)
    res=res.masked_fill(fully_masked,0.0)
    return res

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    res=torch.matmul(attention_weights,value)
    return res

# Step 22 - scaled_dot_product_attention
import math
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_k)

    fully_masked = None

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        fully_masked = torch.isneginf(scores).all(dim=-1, keepdim=True)
        scores = scores.masked_fill(fully_masked, 0.0)

    attention_weights = torch.softmax(scores, dim=-1)

    if fully_masked is not None:
        attention_weights = attention_weights.masked_fill(fully_masked, 0.0)

    context = torch.matmul(attention_weights, value)

    return context, attention_weights

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L,d_model=tensor.size()
    res=torch.reshape(tensor,(B,L,num_heads,d_model//num_heads))
    return res

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    res=torch.transpose(split_tensor,1,2)
    return res

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    b,num_heads,seq_len,d_k=multi_head_tensor.size()
    res=torch.transpose(multi_head_tensor,1,2)
    res=torch.reshape(res,(b,seq_len,num_heads*d_k))
    return res

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    res= x @ torch.transpose(weight,0,1)
    if bias is not None:
        res+=bias
    return res

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    query=apply_linear_projection(x,w_q,b_q)
    key=apply_linear_projection(x,w_k,b_k)
    value=apply_linear_projection(x,w_v,b_v)
    return (query,key,value)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    
    q=split_last_dim_into_heads(q,num_heads)
    q=transpose_heads_before_sequence(q)

    k=split_last_dim_into_heads(k,num_heads)
    k=transpose_heads_before_sequence(k)

    v=split_last_dim_into_heads(v,num_heads)
    v=transpose_heads_before_sequence(v)
    return (q,k,v)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    context,weights=scaled_dot_product_attention(q_h, k_h, v_h, mask)
    return (context,weights)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    res=merge_heads_back_to_model_dim(context)
    res=apply_linear_projection(res, w_o, b_o)
    return res

# Step 31 - assemble_multi_head_attention_forward (not yet solved)
# TODO: implement

# Step 32 - apply_ffn_first_linear_and_relu (not yet solved)
# TODO: implement

# Step 33 - apply_ffn_second_linear (not yet solved)
# TODO: implement

# Step 34 - position_wise_feed_forward_network (not yet solved)
# TODO: implement

# Step 35 - compute_layer_norm_mean_and_variance (not yet solved)
# TODO: implement

# Step 36 - normalize_and_scale_with_gamma_beta (not yet solved)
# TODO: implement

# Step 37 - apply_residual_add_and_norm (not yet solved)
# TODO: implement

# Step 38 - apply_dropout_with_keep_mask (not yet solved)
# TODO: implement

# Step 39 - encoder_layer_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 40 - encoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 41 - assemble_encoder_layer (not yet solved)
# TODO: implement

# Step 42 - stack_encoder_layers (not yet solved)
# TODO: implement

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

