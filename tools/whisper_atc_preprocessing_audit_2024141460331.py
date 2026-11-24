"""Verify 49 HF/OpenAI mel inputs without overwriting frozen audit evidence."""

import numpy as np
import soundfile as sf
import whisper
import whisper_atc_experiment_2024141460331 as common
from transformers import WhisperFeatureExtractor


def main():
    manifest, _ = common.load_inputs()
    extractor = WhisperFeatureExtractor.from_pretrained(
        common.HERE / "models/whisper-large-v3-atcosim", local_files_only=True
    )
    differences = []
    for row in manifest:
        audio, rate = sf.read(common.ROOT / row["audioRelative"], dtype="float32")
        common.validate_audio_span(len(audio), rate)
        hf = extractor(audio, sampling_rate=rate, return_tensors="np").input_features[0]
        native = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=128).numpy()
        if hf.shape != (128, 3000) or native.shape != hf.shape:
            raise ValueError("mel shape mismatch")
        differences.append(float(np.max(np.abs(hf - native))))
    actual = {
        "recordings": len(manifest),
        "maxAbsMelDifference": max(differences),
        "meanOfMaxDifferences": float(np.mean(differences)),
        "hfShape": [128, 3000],
        "officialShape": [128, 3000],
        "comparison": (
            "HF Whisper 128-mel vs OpenAI Whisper log_mel_spectrogram; "
            "native soundfile float32 16k input"
        ),
    }
    if actual["maxAbsMelDifference"] > 1e-5:
        raise ValueError("unexpected preprocessing mismatch")
    output = common.OUT / "preprocessing-audit.json"
    if output.exists():
        saved = common.read_json(output)
        if saved["recordings"] != len(manifest) or saved["maxAbsMelDifference"] > 1e-5:
            raise ValueError("saved preprocessing audit invalid")
        print("Verified existing audit; frozen evidence left unchanged:", actual, flush=True)
    else:
        common.write_json(output, actual)
        print(actual, flush=True)


if __name__ == "__main__":
    main()
