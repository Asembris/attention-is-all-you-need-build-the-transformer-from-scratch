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

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    Q,_,_=project_to_query_key_value(query, w_q, 0.0, w_k, 0.0, w_v, 0.0)
    _,K,_=project_to_query_key_value(key, w_q, 0.0, w_k, 0.0, w_v, 0.0)
    _,_,V=project_to_query_key_value(value, w_q, 0.0, w_k, 0.0, w_v, 0.0)
    Q_h,K_h,V_h=split_qkv_into_heads(Q,K,V, num_heads)
    context,weights=multi_head_scaled_dot_product_attention(Q_h,K_h,V_h,mask)
    new_context=merge_heads_and_project_output(context, w_o, 0.0)
    return new_context

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    res= x @ w1 + b1
    res=torch.relu_(res)
    return res

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    res=hidden @ w2 + b2
    return res

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    first=apply_ffn_first_linear_and_relu(x,w1,b1)
    second= apply_ffn_second_linear(first,w2,b2)
    return second

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    var=torch.var(x,-1,correction=0,keepdim=True)
    mean=torch.mean(x,-1,keepdim=True)
    return (mean,var)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean,variance=compute_layer_norm_mean_and_variance(x)
    x_m=(x-mean)/torch.sqrt(variance+eps)
    y=gamma*x_m+beta
    return y

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    res=residual_input+sublayer_output
    res=normalize_and_scale_with_gamma_beta(res,gamma,beta,eps)
    return res

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    keep_mask=torch.as_tensor(keep_mask,dtype=x.dtype)
    res= x*keep_mask/keep_prob
    return res

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    MHA=assemble_multi_head_attention_forward(x,x,x, w_q, w_k, w_v, w_o, num_heads, mask=src_mask)
    res=apply_residual_add_and_norm(x,MHA,gamma,beta)
    return res

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    res=position_wise_feed_forward_network(x, w1, b1, w2, b2)
    res=apply_residual_add_and_norm(x, res, gamma, beta, eps=1e-5)
    return res

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    context=encoder_layer_self_attention_sublayer(x, layer_params["w_q"], layer_params["w_k"], layer_params["w_v"], layer_params["w_o"], layer_params["attn_gamma"], layer_params["attn_beta"], num_heads, src_mask)
    res=encoder_layer_feed_forward_sublayer(context,layer_params["w1"], layer_params["b1"], layer_params["w2"], layer_params["b2"], layer_params["ffn_gamma"], layer_params["ffn_beta"])
    return res

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    res=x
    for layer_params in encoder_layer_params_list:
        res=assemble_encoder_layer(res, layer_params, num_heads, src_mask)
    return res

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    sub_output=assemble_multi_head_attention_forward(y,y,y, w_q, w_k, w_v, w_o, num_heads,tgt_mask)
    res=apply_residual_add_and_norm(y, sub_output, gamma, beta)
    return res

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    cross=assemble_multi_head_attention_forward(y, encoder_output, encoder_output, w_q, w_k, w_v, w_o, num_heads, mask=None)
    res=apply_residual_add_and_norm(y,cross,gamma,beta)
    return res

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    ffn=position_wise_feed_forward_network(y, w1, b1, w2, b2)
    res=apply_residual_add_and_norm(y,ffn,gamma,beta)
    return res

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    sub_first=decoder_layer_masked_self_attention_sublayer(y, layer_params["w_q_self"], layer_params["w_k_self"],layer_params["w_v_self"], layer_params["w_o_self"], layer_params["self_gamma"], layer_params["self_beta"], num_heads, tgt_mask)
    sub_second=decoder_layer_cross_attention_sublayer(sub_first, encoder_output, layer_params["w_q_cross"], layer_params["w_k_cross"],layer_params["w_v_cross"], layer_params["w_o_cross"], layer_params["cross_gamma"], layer_params["cross_beta"], num_heads, src_mask)
    res=decoder_layer_feed_forward_sublayer(sub_second, layer_params["w1"], layer_params["b1"], layer_params["w2"], layer_params["b2"], layer_params["ffn_gamma"], layer_params["ffn_beta"])
    return res

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    res=y
    for params in decoder_layer_params_list:
        res=assemble_decoder_layer(res, encoder_output, params, num_heads, src_mask, tgt_mask)
    return res

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    res=decoder_output @ torch.transpose(output_projection_weight,-1,-2)
    if output_projection_bias is not None:
        res+=output_projection_bias
    return res

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    res=torch.transpose(token_embedding_weight,-1,-2)
    return res

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    soft_prob=torch.softmax(logits,dim=-1,dtype=logits.dtype)
    res=torch.log(soft_prob)
    return res

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    d_model=model_params["token_embedding"].size(-1)
    src_seq=src_ids.size(-1)
    tgt_seq=tgt_ids.size(-1)
    embed=model_params["token_embedding"]
    src=embed[src_ids]
    tgt=embed[tgt_ids]
    src=scale_embeddings_by_sqrt_d_model(src, d_model)
    tgt=scale_embeddings_by_sqrt_d_model(tgt, d_model)
    pe=build_sinusoidal_positional_encoding(max(src_seq,tgt_seq), d_model)
    src=add_positional_encoding_to_embeddings(src, pe)
    tgt=add_positional_encoding_to_embeddings(tgt, pe)

    src_mask=build_padding_mask(src_ids, pad_id)
    tgt_pad_mask=build_padding_mask(tgt_ids, pad_id)

    tgt_causal_mask=build_causal_mask(tgt_seq)
    tgt_mask=combine_padding_and_causal_masks(tgt_pad_mask,tgt_causal_mask)

    encoder_output=stack_encoder_layers(src,model_params["encoder_layers"], num_heads, src_mask)

    decoder_output=stack_decoder_layers(tgt, encoder_output, model_params["decoder_layers"], num_heads, src_mask, tgt_mask)
    logits=apply_final_output_projection(decoder_output,model_params["output_projection"])
    res=apply_log_softmax_over_vocab(logits)
    return res

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model,num_heads,d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    w_q=torch.randn((d_model,d_model),dtype=torch.float32,requires_grad=True)
    w_k=torch.randn((d_model,d_model),dtype=torch.float32,requires_grad=True)
    w_v=torch.randn((d_model,d_model),dtype=torch.float32,requires_grad=True)
    w_o=torch.randn((d_model,d_model),dtype=torch.float32,requires_grad=True)
    w1=(torch.randn((d_model,d_ff),dtype=torch.float32)/d_model**0.5).requires_grad_()
    b1=torch.zeros((d_ff,),dtype=torch.float32,requires_grad=True)
    w2=(torch.randn((d_ff,d_model),dtype=torch.float32)/d_ff**0.5).requires_grad_()
    b2=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    attn_gamma=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    attn_beta=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    ffn_gamma=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    ffn_beta=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    return {"w_q":w_q,"w_k":w_k,"w_v":w_v,"w_o":w_o,"w1":w1,"b1":b1,"w2":w2,"b2":b2,"attn_gamma":attn_gamma,"attn_beta":attn_beta,"ffn_gamma":ffn_gamma,"ffn_beta":ffn_beta}

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model,num_heads,d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    w_q_self=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_k_self=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_v_self=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_o_self=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_q_cross=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_k_cross=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_v_cross=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w_o_cross=(torch.randn((d_model,d_model),dtype=torch.float32)/d_model**0.5).requires_grad_()
    w1=(torch.randn((d_model,d_ff),dtype=torch.float32)/d_model**0.5).requires_grad_()
    b1=torch.zeros((d_ff,),dtype=torch.float32,requires_grad=True)
    w2=(torch.randn((d_ff,d_model),dtype=torch.float32)/d_ff**0.5).requires_grad_()
    b2=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    self_gamma=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    self_beta=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    cross_gamma=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    cross_beta=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    ffn_gamma=torch.ones((d_model,),dtype=torch.float32,requires_grad=True)
    ffn_beta=torch.zeros((d_model,),dtype=torch.float32,requires_grad=True)
    return {"w_q_self":w_q_self,"w_k_self":w_k_self,"w_v_self":w_v_self,"w_o_self":w_o_self,"w_q_cross":w_q_cross,"w_k_cross":w_k_cross,"w_v_cross":w_v_cross,"w_o_cross":w_o_cross,"w1":w1,"b1":b1,"w2":w2,"b2":b2,"self_gamma":self_gamma,"self_beta":self_beta,"cross_gamma":cross_gamma,"cross_beta":cross_beta,"ffn_gamma":ffn_gamma,"ffn_beta":ffn_beta}

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    src=torch.randn((vocab_size,d_model),dtype=torch.float32,requires_grad=True)
    tgt=torch.randn((vocab_size,d_model),dtype=torch.float32,requires_grad=True)
    if tie_weights:
        output_projection=tgt
    else:
        output_projection=torch.randn((vocab_size,d_model),dtype=torch.float32,requires_grad=True)
    d={"output_projection":output_projection,"src_embedding":src,"tgt_embedding":tgt}
    return d

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params,decoder_layer_params,embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    params_all=[]
    seen=set()
    for layer in encoder_layer_params:
        for param in layer.values():
            if param.requires_grad and id(param) not in seen:
                params_all.append(param)
                seen.add(id(param))
    for layer in decoder_layer_params:
        for param in layer.values():
            if param.requires_grad and id(param) not in seen:
                params_all.append(param)
                seen.add(id(param))
    for param in embedding_params.values():
        if param.requires_grad and id(param) not in seen:
            params_all.append(param)
            seen.add(id(param))
    return params_all

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    b,tgt_seq=target_ids.size()
    start_tokens=torch.full((b,1),start_token_id,dtype=target_ids.dtype,device=target_ids.device)
    res=torch.cat((start_tokens,target_ids[:,:-1]),dim=1)
    return res

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    lr=d_model**(-1/2)*min(step**(-1/2),step*(warmup_steps**(-3/2)))
    return lr

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    const=epsilon/(vocab_size-2)
    res=torch.full(shape,const)
    return res

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    res=smoothed_distribution.scatter(dim=-1,index=gold_token_ids.unsqueeze(-1),value=confidence)
    return res

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    res=smoothed_distribution.clone()
    res[:,:,pad_id]=0
    pad_rows=(gold_token_ids==pad_id).unsqueeze(-1)
    res=res.masked_fill(pad_rows,0)
    return res

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    res=torch.sum(-log_probabilities*smoothed_distribution)
    return res

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    count=torch.sum(gold_token_ids != pad_id)
    count=count.clamp_min(1)
    res=total_loss/count
    return res

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    
    gold_not_pad_ids=gold_token_ids != pad_id
    predicted_ids=torch.argmax(log_probabilities,dim=-1)
    res=torch.sum((predicted_ids==gold_token_ids)&gold_not_pad_ids)/gold_not_pad_ids.sum().clamp_min(1)
    return res

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    m=[torch.zeros_like(param) for param in parameter_list]
    v=[torch.zeros_like(param) for param in parameter_list]
    return {"m":m,"v":v,"t":0}

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    m_t=beta1 *m_prev +(1-beta1)*grad
    return m_t

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    v_t=beta2*v_prev +(1-beta2)*grad*grad
    return v_t

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_t_c=m_t/(1-beta1**step)
    v_t_c=v_t/(1-beta2**step)
    return (m_t_c,v_t_c)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    optimizer_state["t"]+=1
    for i in range(len(parameter_list)):
        grad=parameter_list[i].grad
        if grad is not None:
            optimizer_state["m"][i]=update_adam_first_moment(optimizer_state["m"][i], grad, beta1)
            optimizer_state["v"][i]=update_adam_second_moment(optimizer_state["v"][i], grad, beta2)
            m_c,v_c=apply_adam_bias_correction(optimizer_state["m"][i], optimizer_state["v"][i], beta1, beta2, optimizer_state["t"])
            delta=learning_rate * m_c/(torch.sqrt(v_c)+epsilon)
            with torch.no_grad():
                parameter_list[i]-=delta
    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad=None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch,tgt_batch,model_params,config):
    shifted_tgt=shift_targets_right_with_start_token(tgt_batch,config["start_id"])

    src_embedding=model_params["src_embedding"]
    tgt_embedding=model_params["tgt_embedding"]
    model_params["token_embedding"]=tgt_embedding

    d_model=src_embedding.size(-1)

    src=src_embedding[src_batch]
    tgt=tgt_embedding[shifted_tgt]

    # Keep embeddings connected to autograd; do not call the old helper that uses torch.tensor(...)
    src=src*(d_model**0.5)
    tgt=tgt*(d_model**0.5)

    src_len=src_batch.size(-1)
    tgt_len=shifted_tgt.size(-1)

    pe=build_sinusoidal_positional_encoding(max(src_len,tgt_len),d_model)
    pe=pe.to(device=src.device,dtype=src.dtype)

    src=add_positional_encoding_to_embeddings(src,pe)
    tgt=add_positional_encoding_to_embeddings(tgt,pe)

    src_mask=build_padding_mask(src_batch,config["pad_id"])
    tgt_padding_mask=build_padding_mask(shifted_tgt,config["pad_id"])
    tgt_causal_mask=build_causal_mask(tgt_len).to(shifted_tgt.device)
    tgt_mask=combine_padding_and_causal_masks(tgt_padding_mask,tgt_causal_mask)

    encoder_output=stack_encoder_layers(
        src,
        model_params["encoder_layers"],
        config["num_heads"],
        src_mask
    )

    decoder_output=stack_decoder_layers(
        tgt,
        encoder_output,
        model_params["decoder_layers"],
        config["num_heads"],
        src_mask,
        tgt_mask
    )

    logits=apply_final_output_projection(
        decoder_output,
        model_params["output_projection"]
    )

    log_probs=apply_log_softmax_over_vocab(logits)

    smoothing=config["smoothing"]
    confidence=1.0-smoothing

    smoothed_targets=build_uniform_smoothing_distribution(
        log_probs.shape,
        config["vocab_size"],
        smoothing
    ).to(device=log_probs.device,dtype=log_probs.dtype)

    smoothed_targets=set_confidence_on_gold_tokens(
        smoothed_targets,
        tgt_batch,
        confidence
    )

    smoothed_targets=zero_pad_column_and_pad_token_rows(
        smoothed_targets,
        tgt_batch,
        config["pad_id"]
    )

    total_loss=compute_label_smoothed_kl_loss(
        log_probs,
        smoothed_targets
    )

    loss=average_loss_over_non_pad_tokens(
        total_loss,
        tgt_batch,
        config["pad_id"]
    )

    return loss

# Step 72 - run_training_step_with_backprop
import torch

def run_training_step_with_backprop(src_batch,tgt_batch,parameter_list,model_params,optimizer_state,step_number,config):
    """Run one training iteration: zero grads, forward, backward, Noam LR, Adam step.

    Returns the scalar loss value for the step as a Python float.
    """
    zero_all_parameter_gradients(parameter_list)

    loss=compute_batch_training_loss(
        src_batch,
        tgt_batch,
        model_params,
        config
    )

    loss.backward()

    learning_rate=compute_noam_learning_rate(
        step_number,
        config["d_model"],
        config["warmup_steps"]
    )

    apply_adam_step_to_all_parameters(
        parameter_list,
        optimizer_state,
        learning_rate,
        config.get("beta1",0.9),
        config.get("beta2",0.98),
        config.get("epsilon",1e-9)
    )

    return loss.item()

# Step 73 - run_training_loop_for_steps
def run_training_loop_for_steps(batches,parameter_list,model_params,optimizer_state,num_steps,config):
    """Run num_steps training iterations, cycling through batches, and return per-step losses."""
    losses=[]
    for i in range(1,num_steps+1):
        src_batch,tgt_batch=batches[(i-1)%len(batches)]
        loss=run_training_step_with_backprop(src_batch,tgt_batch,parameter_list,model_params,optimizer_state,i,config)
        losses.append(loss)
    return losses

# Step 74 - pick_next_token_by_argmax
import torch

def pick_next_token_by_argmax(final_step_logits):
    """Greedy: return argmax token id per batch row.

    final_step_logits: FloatTensor of shape (batch, vocab_size)
    returns: LongTensor of shape (batch,)
    """
    # TODO: pick the next greedy token id by taking the argmax over the vocab axis
    return torch.argmax(final_step_logits,dim=-1)

# Step 75 - compute_length_penalty
def compute_length_penalty(sequence_length, alpha):
    # TODO: return the Google NMT length penalty for the given sequence_length and alpha.
    return ((5+sequence_length)/6)**alpha

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

