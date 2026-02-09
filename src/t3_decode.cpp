// C++ T3 autoregressive decode loop for chatterbox
//
// Eliminates R→lantern→libtorch per-op dispatch overhead by running the
// entire decode loop in a single .Call(). Prefill stays in R (one call,
// no loop overhead).
//
// IMPORTANT: Include torch headers before R headers to avoid type conflicts.

#include <torch/torch.h>
#include <ATen/Functions.h>
#include <cmath>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <R.h>
#include <Rinternals.h>

// ============================================================================
// Tensor access helpers (same pattern as torchlang/src/fused_kernels.cpp)
// ============================================================================

static torch::Tensor* get_tensor_ptr(SEXP x) {
    if (TYPEOF(x) != EXTPTRSXP) return nullptr;
    void* xptr = R_ExternalPtrAddr(x);
    if (!xptr) return nullptr;
    auto* sptr = static_cast<std::shared_ptr<void>*>(xptr);
    return static_cast<torch::Tensor*>(sptr->get());
}

// Get a tensor from a named R list element
static torch::Tensor get_tensor_from_list(SEXP list, int idx) {
    SEXP elem = VECTOR_ELT(list, idx);
    torch::Tensor* ptr = get_tensor_ptr(elem);
    if (!ptr) Rf_error("Invalid tensor at list index %d", idx);
    return *ptr;
}

// Get a double from a named R list
static double get_double(SEXP list, const char* name) {
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    for (int i = 0; i < Rf_length(list); i++) {
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
            return Rf_asReal(VECTOR_ELT(list, i));
        }
    }
    Rf_error("Missing config parameter: %s", name);
    return 0.0;
}

// Get an integer from a named R list
static int64_t get_int(SEXP list, const char* name) {
    SEXP names = Rf_getAttrib(list, R_NamesSymbol);
    for (int i = 0; i < Rf_length(list); i++) {
        if (strcmp(CHAR(STRING_ELT(names, i)), name) == 0) {
            return (int64_t)Rf_asInteger(VECTOR_ELT(list, i));
        }
    }
    Rf_error("Missing config parameter: %s", name);
    return 0;
}

// ============================================================================
// Model operations (matching R/llama.R implementations)
// ============================================================================

// RMSNorm (matches llama_rms_norm$forward in R/llama.R:65-72)
static torch::Tensor rms_norm(const torch::Tensor& x, const torch::Tensor& weight, float eps) {
    auto x_f32 = x.to(torch::kFloat32);
    auto variance = x_f32.pow(2).mean(-1, /*keepdim=*/true);
    auto normed = x_f32 * torch::rsqrt(variance + eps);
    return weight * normed.to(x.dtype());
}

// Rotate half for RoPE (matches rotate_half in R/llama.R:158-163)
static torch::Tensor rotate_half(const torch::Tensor& x) {
    int64_t half = x.size(-1) / 2;
    auto x1 = x.slice(-1, 0, half);
    auto x2 = x.slice(-1, half);
    return torch::cat({-x2, x1}, -1);
}

// Apply RoPE to Q and K for a single position
// cos_pos, sin_pos: already sliced to current position shape (1, 1, dim)
static void apply_rope_single(torch::Tensor& q, torch::Tensor& k,
                               const torch::Tensor& cos_pos,
                               const torch::Tensor& sin_pos) {
    q = q * cos_pos + rotate_half(q) * sin_pos;
    k = k * cos_pos + rotate_half(k) * sin_pos;
}

// Layer weights structure for a single decoder layer
struct LayerWeights {
    torch::Tensor input_ln_weight;     // RMSNorm weight
    torch::Tensor q_proj_weight;       // (n_heads * head_dim, hidden_size)
    torch::Tensor k_proj_weight;
    torch::Tensor v_proj_weight;
    torch::Tensor o_proj_weight;
    torch::Tensor post_ln_weight;      // RMSNorm weight
    torch::Tensor gate_proj_weight;    // (intermediate_size, hidden_size)
    torch::Tensor up_proj_weight;
    torch::Tensor down_proj_weight;    // (hidden_size, intermediate_size)
};

// Single decoder layer forward pass (matches llama_decoder_layer$forward)
// hidden_states: (B, 1, hidden_size)
// Returns updated hidden_states
static torch::Tensor decoder_layer_forward(
    const torch::Tensor& hidden_states,
    const LayerWeights& w,
    torch::Tensor& k_cache,       // (B, n_heads, max_len, head_dim) — mutable
    torch::Tensor& v_cache,       // (B, n_heads, max_len, head_dim) — mutable
    int64_t current_pos,          // 0-indexed position to write K/V
    int64_t valid_len,            // number of valid positions (for attention)
    const torch::Tensor& cos_pos, // (1, 1, head_dim) for current position
    const torch::Tensor& sin_pos,
    int64_t n_heads,
    int64_t head_dim,
    float rms_eps
) {
    int64_t bsz = hidden_states.size(0);
    int64_t hidden_size = hidden_states.size(2);

    // === Self-attention ===
    auto residual = hidden_states;

    // Pre-norm
    auto normed = rms_norm(hidden_states, w.input_ln_weight, rms_eps);

    // Q/K/V projections: (B, 1, hidden) → (B, 1, n_heads*head_dim)
    auto q = torch::linear(normed, w.q_proj_weight);
    auto k = torch::linear(normed, w.k_proj_weight);
    auto v = torch::linear(normed, w.v_proj_weight);

    // Reshape to (B, n_heads, 1, head_dim)
    q = q.view({bsz, 1, n_heads, head_dim}).transpose(1, 2);
    k = k.view({bsz, 1, n_heads, head_dim}).transpose(1, 2);
    v = v.view({bsz, 1, n_heads, head_dim}).transpose(1, 2);

    // Apply RoPE
    apply_rope_single(q, k, cos_pos, sin_pos);

    // Write K/V to cache at current_pos
    k_cache.select(2, current_pos).copy_(k.squeeze(2));
    v_cache.select(2, current_pos).copy_(v.squeeze(2));

    // Slice valid K/V from cache: (B, n_heads, valid_len, head_dim)
    auto k_valid = k_cache.narrow(2, 0, valid_len);
    auto v_valid = v_cache.narrow(2, 0, valid_len);

    // SDPA — at::scaled_dot_product_attention
    auto attn_out = at::scaled_dot_product_attention(
        q, k_valid, v_valid,
        /*attn_mask=*/{},      // no mask needed — all valid_len positions are valid
        /*dropout_p=*/0.0,
        /*is_causal=*/false    // causal not needed for single query token attending to past
    );

    // Reshape back: (B, n_heads, 1, head_dim) → (B, 1, hidden)
    attn_out = attn_out.transpose(1, 2).contiguous().view({bsz, 1, hidden_size});

    // O projection + residual
    auto h = torch::linear(attn_out, w.o_proj_weight) + residual;

    // === MLP (SwiGLU) ===
    residual = h;
    auto h_norm = rms_norm(h, w.post_ln_weight, rms_eps);

    // gate_proj and up_proj
    auto gate = torch::linear(h_norm, w.gate_proj_weight);
    auto up = torch::linear(h_norm, w.up_proj_weight);

    // SiLU(gate) * up → down_proj
    auto mlp_out = torch::linear(torch::silu(gate) * up, w.down_proj_weight);

    return residual + mlp_out;
}

// ============================================================================
// Main decode loop
// ============================================================================

extern "C" {

// cpp_t3_decode: Run the entire autoregressive decode loop in C++
//
// Arguments (all SEXP):
//   layer_weights    - R list of 30 sublists, each with 9 weight tensors:
//                      [input_ln, q_proj, k_proj, v_proj, o_proj,
//                       post_ln, gate_proj, up_proj, down_proj]
//   final_norm_weight - final RMSNorm weight tensor
//   speech_head_weight - speech head linear weight (vocab_size, hidden_size)
//   speech_emb_weight  - speech embedding weight (vocab_size, hidden_size)
//   speech_pos_emb_weight - speech position embedding weight (max_speech_tokens, hidden_size)
//   k_cache          - (n_layers, B, n_heads, max_len, head_dim) pre-filled from prefill
//   v_cache          - (n_layers, B, n_heads, max_len, head_dim)
//   initial_hidden   - (B, 1, hidden_size) — ALREADY NORMED (from llama_model$forward)
//   rope_cos         - (max_len, head_dim) pre-computed
//   rope_sin         - (max_len, head_dim)
//   config           - R list with decode parameters
//
// Returns: integer vector of generated 0-indexed token IDs

SEXP cpp_t3_decode(
    SEXP layer_weights_sexp,
    SEXP final_norm_weight_sexp,
    SEXP speech_head_weight_sexp,
    SEXP speech_emb_weight_sexp,
    SEXP speech_pos_emb_weight_sexp,
    SEXP k_cache_sexp,
    SEXP v_cache_sexp,
    SEXP initial_hidden_sexp,
    SEXP rope_cos_sexp,
    SEXP rope_sin_sexp,
    SEXP config_sexp
) {
    // === Extract config ===
    int64_t cond_len = get_int(config_sexp, "cond_len");
    int64_t max_tokens = get_int(config_sexp, "max_tokens");
    double temperature = get_double(config_sexp, "temperature");
    double cfg_weight = get_double(config_sexp, "cfg_weight");
    double top_p = get_double(config_sexp, "top_p");
    double min_p = get_double(config_sexp, "min_p");
    double rep_penalty = get_double(config_sexp, "rep_penalty");
    int64_t stop_token_0idx = get_int(config_sexp, "stop_token");
    int64_t n_heads = get_int(config_sexp, "n_heads");
    int64_t head_dim = get_int(config_sexp, "head_dim");
    float rms_eps = (float)get_double(config_sexp, "rms_eps");

    // === Extract tensors ===
    torch::Tensor final_norm_w = *get_tensor_ptr(final_norm_weight_sexp);
    torch::Tensor speech_head_w = *get_tensor_ptr(speech_head_weight_sexp);
    torch::Tensor speech_emb_w = *get_tensor_ptr(speech_emb_weight_sexp);
    torch::Tensor speech_pos_emb_w = *get_tensor_ptr(speech_pos_emb_weight_sexp);
    torch::Tensor rope_cos = *get_tensor_ptr(rope_cos_sexp);
    torch::Tensor rope_sin = *get_tensor_ptr(rope_sin_sexp);

    // K/V caches: (n_layers, B, n_heads, max_len, head_dim)
    torch::Tensor k_cache_all = *get_tensor_ptr(k_cache_sexp);
    torch::Tensor v_cache_all = *get_tensor_ptr(v_cache_sexp);

    // Initial hidden state from prefill (already normed by llama_model$forward)
    torch::Tensor normed_hidden = *get_tensor_ptr(initial_hidden_sexp);

    int64_t n_layers = Rf_length(layer_weights_sexp);
    int64_t batch_size = normed_hidden.size(0);  // 2 for CFG, 1 otherwise
    int64_t hidden_size = normed_hidden.size(2);
    int64_t max_cache_len = k_cache_all.size(3);
    int64_t vocab_size = speech_head_w.size(0);
    bool use_cfg = (cfg_weight > 0.0 && batch_size > 1);

    // === Extract layer weights ===
    std::vector<LayerWeights> layers(n_layers);
    for (int64_t i = 0; i < n_layers; i++) {
        SEXP lw = VECTOR_ELT(layer_weights_sexp, i);
        layers[i].input_ln_weight   = get_tensor_from_list(lw, 0);
        layers[i].q_proj_weight     = get_tensor_from_list(lw, 1);
        layers[i].k_proj_weight     = get_tensor_from_list(lw, 2);
        layers[i].v_proj_weight     = get_tensor_from_list(lw, 3);
        layers[i].o_proj_weight     = get_tensor_from_list(lw, 4);
        layers[i].post_ln_weight    = get_tensor_from_list(lw, 5);
        layers[i].gate_proj_weight  = get_tensor_from_list(lw, 6);
        layers[i].up_proj_weight    = get_tensor_from_list(lw, 7);
        layers[i].down_proj_weight  = get_tensor_from_list(lw, 8);
    }

    // === Per-layer KV cache views (avoid repeated slicing) ===
    std::vector<torch::Tensor> k_caches(n_layers);
    std::vector<torch::Tensor> v_caches(n_layers);
    for (int64_t i = 0; i < n_layers; i++) {
        k_caches[i] = k_cache_all.select(0, i);  // (B, n_heads, max_len, head_dim)
        v_caches[i] = v_cache_all.select(0, i);
    }

    // === Decode loop state ===
    // In C++ we work with 0-indexed token IDs throughout.
    std::vector<int64_t> generated_tokens;
    generated_tokens.reserve(max_tokens);

    // Track unique generated token IDs for repetition penalty
    std::unordered_set<int64_t> seen_tokens;

    auto device = normed_hidden.device();

    torch::NoGradGuard no_grad;

    // The loop structure matches the R code:
    //   1. Apply speech_head to normed hidden → logits
    //   2. Sample token
    //   3. Compute new embedding
    //   4. Run through decoder layers → un-normed hidden
    //   5. Apply final norm → normed hidden
    //   6. Go to 1
    //
    // On first iteration, normed_hidden comes from prefill (already normed).
    // On subsequent iterations, we norm at end of step.

    for (int64_t step = 0; step < max_tokens; step++) {
        // === Speech head: normed hidden → logits ===
        // normed_hidden: (B, 1, hidden_size) — already normed
        auto logits = torch::linear(normed_hidden.squeeze(1), speech_head_w);
        // logits: (B, vocab_size)

        // === CFG combination ===
        torch::Tensor combined_logits;
        if (use_cfg) {
            auto cond_logits = logits.select(0, 0);    // (vocab_size,)
            auto uncond_logits = logits.select(0, 1);
            combined_logits = cond_logits + cfg_weight * (cond_logits - uncond_logits);
        } else {
            combined_logits = logits.select(0, 0);
        }
        // Shape: (vocab_size,) — work with 1D from here

        // === Repetition penalty ===
        // The R code divides by rep_penalty for all penalized tokens.
        // We match that behavior (simpler than sign-dependent penalty).
        if (rep_penalty != 1.0 && !seen_tokens.empty()) {
            for (int64_t tok : seen_tokens) {
                if (tok >= 0 && tok < vocab_size) {
                    combined_logits[tok] = combined_logits[tok] / rep_penalty;
                }
            }
        }

        // === Temperature ===
        if (temperature != 1.0) {
            combined_logits = combined_logits / temperature;
        }

        // === Softmax → probabilities ===
        auto probs = torch::softmax(combined_logits, 0);

        // === Min-p filtering ===
        {
            float max_prob = probs.max().item<float>();
            float min_threshold = (float)min_p * max_prob;
            probs = torch::where(probs >= min_threshold, probs, torch::zeros_like(probs));
            probs = probs / probs.sum();
        }

        // === Top-p (nucleus) sampling ===
        auto [sorted_probs, sorted_indices] = torch::sort(probs, /*dim=*/0, /*descending=*/true);
        auto cumsum = torch::cumsum(sorted_probs, 0);

        // Zero out tokens beyond top_p threshold (keep at least first)
        auto mask = cumsum > top_p;
        // Shift mask right: don't zero out the token that first crosses threshold
        auto shifted_mask = torch::zeros_like(mask);
        if (mask.size(0) > 1) {
            shifted_mask.slice(0, 1) = mask.slice(0, 0, mask.size(0) - 1);
        }
        sorted_probs = torch::where(shifted_mask, torch::zeros_like(sorted_probs), sorted_probs);
        sorted_probs = sorted_probs / sorted_probs.sum();

        // === Multinomial sample ===
        auto token_idx = torch::multinomial(sorted_probs, 1);
        int64_t sampled_sort_idx = token_idx.item<int64_t>();
        int64_t token_id = sorted_indices[sampled_sort_idx].item<int64_t>();
        // token_id is 0-indexed (C++ tensor indexing)

        generated_tokens.push_back(token_id);
        seen_tokens.insert(token_id);

        // === EOS check ===
        if (token_id == stop_token_0idx) {
            break;
        }

        // === Embedding lookup for next token ===
        // speech_emb_w: (vocab_size, hidden_size) — 0-indexed in C++
        auto token_emb = speech_emb_w.select(0, token_id).unsqueeze(0).unsqueeze(0);
        // token_emb: (1, 1, hidden_size)

        // Position embedding: step+1 because step 0 was the BOS token
        // speech_pos_emb_w: (max_speech_tokens, hidden_size) — 0-indexed in C++
        auto pos_emb = speech_pos_emb_w.select(0, step + 1).unsqueeze(0).unsqueeze(0);

        auto next_emb = token_emb + pos_emb;

        // Double for CFG
        if (use_cfg) {
            next_emb = torch::cat({next_emb, next_emb}, 0);
            // next_emb: (2, 1, hidden_size)
        }

        // === Current position in KV cache ===
        int64_t current_pos = cond_len + step;  // 0-indexed
        int64_t valid_len = current_pos + 1;     // how many positions are valid

        if (valid_len > max_cache_len) {
            Rf_warning("KV cache full at step %lld, stopping generation", (long long)step);
            break;
        }

        // RoPE for current position: (1, 1, head_dim)
        auto cos_pos = rope_cos.select(0, current_pos).unsqueeze(0).unsqueeze(0);
        auto sin_pos = rope_sin.select(0, current_pos).unsqueeze(0).unsqueeze(0);

        // === Run through all decoder layers ===
        auto hidden = next_emb;
        for (int64_t l = 0; l < n_layers; l++) {
            hidden = decoder_layer_forward(
                hidden, layers[l],
                k_caches[l], v_caches[l],
                current_pos, valid_len,
                cos_pos, sin_pos,
                n_heads, head_dim, rms_eps
            );
        }

        // === Apply final norm (so next iteration can apply speech_head directly) ===
        normed_hidden = rms_norm(hidden, final_norm_w, rms_eps);
    }

    // === Return generated token IDs as R integer vector ===
    int n_tokens = (int)generated_tokens.size();
    SEXP result = PROTECT(Rf_allocVector(INTSXP, n_tokens));
    int* result_ptr = INTEGER(result);
    for (int i = 0; i < n_tokens; i++) {
        result_ptr[i] = (int)generated_tokens[i];
    }
    UNPROTECT(1);
    return result;
}

} // extern "C"
