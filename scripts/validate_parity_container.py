# Container reference for the same validation cases (Python 0.1.4).
import time
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

CASES = [
    ("control_hello", "Hello world, this is a test of the chatterbox text to speech system"),
    ("issue2_mixedcase", "Yes, Rarely or never Almost never, at most once in a while, over the past week Sometimes Only a couple of days over the past week, not many times in any given day Often Four or more days over the past week, several times each day Very Often just about every day over the past week, multiple times throughout the Day."),
    ("issue1_emphasis", "Very often. As I said earlier, homework is a daily battle. He will do anything to get out of it. My head hurts; My stomach hurts. He asks for snacks or stops working to get a glass of water."),
    ("issue1_long", "This would be moderate problem because both his teacher and I spend the time to organize Harry and make sure he has taken home his assignments and returned them to school, but no plan is foolproof."),
    ("paraling_laughter", "Well that is just wonderful news. [laughter] I could not be happier for you."),
]

model = ChatterboxTTS.from_pretrained(device="cuda")
ref = "/ref/jfk.wav"

for label, text in CASES:
    t0 = time.time()
    wav = model.generate(text, audio_prompt_path=ref)
    dt = time.time() - t0
    out = f"/out/{label}.wav"
    ta.save(out, wav, model.sr)
    print(f"{label}: {wav.shape[1]/model.sr:.2f}s audio, std={wav.std().item():.4f}, gen={dt:.1f}s", flush=True)
