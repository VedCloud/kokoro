import soundfile as sf
from kokoro import KPipeline

pipeline = KPipeline(lang_code='d')
text = "Kokoro German ist ein Text to Speech Modell mit zweiundachtzig Millionen Parametern."

# Force the framework to evaluate our newly placed voice file off local disk
generator = pipeline(text, voice='voices/df_kerstin.pt', speed=1.0)
for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f"german_test_output_{i}.wav", audio, 24000)
    print(f"Generated audio block {i} successfully using local German tensors.")
