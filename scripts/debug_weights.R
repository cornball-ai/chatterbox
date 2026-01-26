#!/usr/bin/env Rscript
# Debug weight loading

library(torch)
devtools::load_all("/home/troy/chatterbox")

# Load weights
weights_path <- Sys.glob(path.expand("~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/*/s3gen.safetensors"))[1]
state_dict <- read_safetensors(weights_path, device = "cpu")

# Create encoder
encoder <- upsample_conformer_encoder_full()
torch::with_no_grad({
        load_conformer_encoder_weights(encoder, state_dict, prefix = "flow.encoder.")
    })

cat("=== Check conformer block 0 weights ===\n")

# Check self_attn weights
check_weight <- function (r_param, key)
{
    full_key <- paste0("flow.encoder.", key)
    if (full_key %in% names(state_dict)) {
        py_w <- state_dict[[full_key]]
        diff <- (r_param - py_w)$abs()$max()$item()
        cat(sprintf("  %s: diff=%.6f\n", key, diff))
        return(diff)
    } else {
        cat(sprintf("  %s: KEY NOT FOUND\n", key))
        return(NA)
    }
}

# Block 0 attention
cat("\nBlock 0 self_attn:\n")
check_weight(encoder$encoders[[1]]$self_attn$linear_q$weight, "encoders.0.self_attn.linear_q.weight")
check_weight(encoder$encoders[[1]]$self_attn$linear_q$bias, "encoders.0.self_attn.linear_q.bias")
check_weight(encoder$encoders[[1]]$self_attn$linear_k$weight, "encoders.0.self_attn.linear_k.weight")
check_weight(encoder$encoders[[1]]$self_attn$linear_v$weight, "encoders.0.self_attn.linear_v.weight")
check_weight(encoder$encoders[[1]]$self_attn$linear_out$weight, "encoders.0.self_attn.linear_out.weight")
check_weight(encoder$encoders[[1]]$self_attn$linear_pos$weight, "encoders.0.self_attn.linear_pos.weight")
check_weight(encoder$encoders[[1]]$self_attn$pos_bias_u, "encoders.0.self_attn.pos_bias_u")
check_weight(encoder$encoders[[1]]$self_attn$pos_bias_v, "encoders.0.self_attn.pos_bias_v")

# Block 0 feed_forward
cat("\nBlock 0 feed_forward:\n")
check_weight(encoder$encoders[[1]]$feed_forward$w_1$weight, "encoders.0.feed_forward.w_1.weight")
check_weight(encoder$encoders[[1]]$feed_forward$w_1$bias, "encoders.0.feed_forward.w_1.bias")
check_weight(encoder$encoders[[1]]$feed_forward$w_2$weight, "encoders.0.feed_forward.w_2.weight")
check_weight(encoder$encoders[[1]]$feed_forward$w_2$bias, "encoders.0.feed_forward.w_2.bias")

# Block 0 layer norms
cat("\nBlock 0 layer norms:\n")
check_weight(encoder$encoders[[1]]$norm_mha$weight, "encoders.0.norm_mha.weight")
check_weight(encoder$encoders[[1]]$norm_mha$bias, "encoders.0.norm_mha.bias")
check_weight(encoder$encoders[[1]]$norm_ff$weight, "encoders.0.norm_ff.weight")
check_weight(encoder$encoders[[1]]$norm_ff$bias, "encoders.0.norm_ff.bias")

# Check up_embed
cat("\n=== Check up_embed weights ===\n")
check_weight(encoder$up_embed$out[[1]]$weight, "up_embed.out.0.weight")
check_weight(encoder$up_embed$out[[1]]$bias, "up_embed.out.0.bias")
check_weight(encoder$up_embed$out[[2]]$weight, "up_embed.out.1.weight")
check_weight(encoder$up_embed$out[[2]]$bias, "up_embed.out.1.bias")

# Check embed
cat("\n=== Check embed weights ===\n")
check_weight(encoder$embed$out[[1]]$weight, "embed.out.0.weight")
check_weight(encoder$embed$out[[1]]$bias, "embed.out.0.bias")
check_weight(encoder$embed$out[[2]]$weight, "embed.out.1.weight")
check_weight(encoder$embed$out[[2]]$bias, "embed.out.1.bias")

cat("\n=== Done ===\n")

