"""Reference outputs for chatterbox R ports of torchaudio resample + kaldi fbank.

Reads /tmp/test_signal.csv (one sample per line, 24000 Hz), computes:
  (a) torchaudio.transforms.Resample(24000, 16000) -> /tmp/ref_resample.csv
  (b) torchaudio.compliance.kaldi.fbank(resampled, num_mel_bins=80)
      -> /tmp/ref_fbank.csv (row-major flat)

Run inside the reference container:
  docker run --rm -v /tmp:/tmp -v /home/troy/chatterbox/scripts:/scripts \
    chatterbox-tts:blackwell python /scripts/save_kaldi_resample_ref.py
"""
import torch
import torchaudio
from torchaudio.compliance import kaldi


def main():
    with open("/tmp/test_signal.csv") as f:
        samples = [float(line) for line in f if line.strip()]
    wav = torch.tensor(samples, dtype=torch.float32)
    print(f"signal: {wav.numel()} samples @ 24000 Hz")

    # (a) resample 24000 -> 16000 with transform defaults
    resampler = torchaudio.transforms.Resample(24000, 16000)
    resampled = resampler(wav.unsqueeze(0)).squeeze(0)
    print(f"resampled: {resampled.numel()} samples @ 16000 Hz")
    with open("/tmp/ref_resample.csv", "w") as f:
        for v in resampled.tolist():
            f.write("%.10g\n" % v)

    # (b) kaldi fbank of the resampled signal (CAMPPlus call signature)
    feat = kaldi.fbank(resampled.unsqueeze(0), num_mel_bins=80)
    print(f"fbank: {tuple(feat.shape)}")
    with open("/tmp/ref_fbank.csv", "w") as f:
        for v in feat.flatten().tolist():
            f.write("%.10g\n" % v)

    print("wrote /tmp/ref_resample.csv and /tmp/ref_fbank.csv")


if __name__ == "__main__":
    main()
