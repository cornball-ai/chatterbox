# Profile pure-R vs traced T3 generation with profvis + debrief.
# Where does per-token wall time actually live: sync stalls ($item/$cpu),
# op-dispatch overhead (spread across torch wrappers), or GC pressure?

library(chatterbox)
library(profvis)
library(debrief)

text <- "Hello world, this is a test of the chatterbox text to speech system"
ref_audio <- system.file("audio", "jfk.wav", package = "chatterbox")

model <- chatterbox(device = "cuda")
model <- load_chatterbox(model)
voice <- create_voice_embedding(model, ref_audio)

# Profile pure-R generation
cat("\n================ PROFILING pure-R ================\n")
p_r <- profvis(
    generate(model, text, voice),
    interval = 0.01, simplify = FALSE
)
saveRDS(p_r, "/tmp/prof_pure_r.rds")

# Warm the traced graphs (cold compile, unprofiled)
cat("\n(warming traced graphs...)\n")
invisible(generate(model, text, voice, traced = TRUE))

# Profile a warm traced generation
cat("\n================ PROFILING traced (warm) ================\n")
p_t <- profvis(
    generate(model, text, voice, traced = TRUE),
    interval = 0.005, simplify = FALSE
)
saveRDS(p_t, "/tmp/prof_traced.rds")

show <- function (label, p) {
    cat("\n========", label, "========\n")
    cat("\n-- self time --\n")
    print(pv_self_time(p, n = 12))
    cat("\n-- hot lines --\n")
    print(pv_hot_lines(p, n = 12))
    cat("\n-- gc pressure --\n")
    print(pv_gc_pressure(p))
    cat("\n-- hot paths --\n")
    print(pv_hot_paths(p, n = 5))
}

show("PURE R", p_r)
show("TRACED WARM", p_t)

cat("\n======== COMPARE (before = pure R, after = traced) ========\n")
print(pv_compare(p_r, p_t, n = 15))
