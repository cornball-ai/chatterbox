# .sniff_audio_format() detects the real container from magic bytes and
# ignores the file extension: a WAV saved with a .mp3 name (which once ran
# the MP3 decoder on PCM bytes and produced NaN garbage) must read as wav.

wav <- system.file("audio", "jfk.wav", package = "chatterbox")
mp3 <- system.file("audio", "jfk.mp3", package = "chatterbox")

# Correctly-named files
expect_equal(chatterbox:::.sniff_audio_format(wav), "wav")
expect_equal(chatterbox:::.sniff_audio_format(mp3), "mp3")

# Extension lies: WAV bytes under a .mp3 name still sniff as wav
fake_mp3 <- tempfile(fileext = ".mp3")
file.copy(wav, fake_mp3)
expect_equal(chatterbox:::.sniff_audio_format(fake_mp3), "wav")
unlink(fake_mp3)

# ... and MP3 bytes under a .wav name still sniff as mp3
fake_wav <- tempfile(fileext = ".wav")
file.copy(mp3, fake_wav)
expect_equal(chatterbox:::.sniff_audio_format(fake_wav), "mp3")
unlink(fake_wav)
