"""Bounded OpenAI-decoder cross-check using the downloaded HF weights.

No base model is downloaded. Strict weight loading and tied-output equality
prevent a partially loaded or mismatched architecture from passing the audit.
"""

import ast
import gc
import io
import time
from pathlib import Path

import whisper_atc_experiment_2024141460331 as common


def native_key(key):
    if key == "proj_out.weight":
        return None
    if not key.startswith("model."):
        raise ValueError(f"unexpected checkpoint key: {key}")
    key = key.removeprefix("model.")
    for before, after in (
        (".layers.", ".blocks."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".self_attn.", ".attn."),
        (".encoder_attn.", ".cross_attn."),
        (".q_proj.", ".query."),
        (".k_proj.", ".key."),
        (".v_proj.", ".value."),
        (".out_proj.", ".out."),
        (".fc1.", ".mlp.0."),
        (".fc2.", ".mlp.2."),
        (".embed_positions.weight", ".positional_embedding"),
        (".embed_tokens.", ".token_embedding."),
        ("encoder.layer_norm.", "encoder.ln_post."),
        ("decoder.layer_norm.", "decoder.ln."),
    ):
        key = key.replace(before, after)
    return key


def load_native():
    import torch
    import whisper
    from safetensors.torch import load_file

    info = common.MODELS["mixed"]
    local = common.HERE / "models" / info["id"].split("/")[-1]
    if common.sha256(local / "model.safetensors") != info["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    cfg = common.read_json(local / "config.json")
    dims = whisper.ModelDimensions(
        n_mels=cfg["num_mel_bins"],
        n_audio_ctx=cfg["max_source_positions"],
        n_audio_state=cfg["d_model"],
        n_audio_head=cfg["encoder_attention_heads"],
        n_audio_layer=cfg["encoder_layers"],
        n_vocab=cfg["vocab_size"],
        n_text_ctx=cfg["max_target_positions"],
        n_text_state=cfg["d_model"],
        n_text_head=cfg["decoder_attention_heads"],
        n_text_layer=cfg["decoder_layers"],
    )
    state = load_file(local / "model.safetensors", device="cpu")
    if not torch.equal(state["proj_out.weight"], state["model.decoder.embed_tokens.weight"]):
        raise ValueError("native output tying would change checkpoint weights")
    mapped = {native_key(k): v for k, v in state.items() if native_key(k) is not None}
    if len(mapped) != len(state) - 1:
        raise ValueError("weight mapping collision")
    saved_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        model = whisper.Whisper(dims)
    finally:
        torch.set_default_dtype(saved_dtype)
    model.load_state_dict(mapped, strict=True)
    # OpenAI LayerNorm casts activations to FP32 and needs FP32 norm weights.
    for layer in model.modules():
        if isinstance(layer, whisper.model.LayerNorm):
            layer.float()
    del mapped, state
    gc.collect()
    return model.to("cuda").eval(), dims.__dict__


def author_prompt(row):
    source = common.RESEARCH / "Evaluate/Evaluate-FineTuned.py"
    literals = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("nato", "terminology"):
                    literals[target.id] = ast.literal_eval(node.value)
    if set(literals) != {"nato", "terminology"}:
        raise ValueError("author vocabulary literals missing")
    return (
        "Air Traffic Control Communications "
        + row["info"].replace("\n", " ")
        + " "
        + literals["nato"].replace(",", " ")
        + " "
        + literals["terminology"].replace(",", " ")
    )


def main():
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import whisper

    output = common.OUT / "native-decoder-audit.json"
    if output.exists():
        raise ValueError("audit already exists; preserve the original record")
    model, dims = load_native()
    manifest, _ = common.load_inputs()
    prior = common.read_jsonl(common.OUT / "whisper-mixed-results.jsonl")
    probes = []
    airports = set()
    for row in manifest:
        if row["airport"] in airports:
            continue
        airports.add(row["airport"])
        old = next(
            x for x in prior if x["recordingId"] == row["recordingId"] and x["condition"] == "xml"
        )["segments"][0]
        audio, rate = sf.read(common.ROOT / row["audioRelative"], dtype="float32")
        audio = audio[round(old["startMs"] * rate / 1000) : round(old["endMs"] * rate / 1000)]
        probes.append(
            {
                "id": row["recordingId"],
                "selection": "first XML segment per airport, fixed manifest order",
                "audio": audio,
                "hfHypothesis": old["textRaw"],
                "prompt": None,
            }
        )
    parquet = next((common.RESEARCH / "published-atco2-dataset/data").glob("validation-*.parquet"))
    first = pq.read_table(parquet).to_pylist()[0]
    audio, rate = sf.read(io.BytesIO(first["audio"]["bytes"]), dtype="float32")
    validation = common.read_jsonl(common.OUT / "published-validation-113/results.jsonl")
    for condition in ("no_prompt", "metadata_prompt"):
        rid = Path(first["audio"]["path"]).stem
        probes.append(
            {
                "id": rid,
                "selection": "first published validation file: " + condition,
                "audio": audio,
                "hfHypothesis": next(
                    x["hypothesis"]
                    for x in validation
                    if x["recordingId"] == rid and x["condition"] == condition
                ),
                "prompt": author_prompt(first) if condition == "metadata_prompt" else None,
            }
        )
    results = []
    for index, probe in enumerate(probes):
        common.validate_audio_span(len(probe["audio"]), 16000)
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(probe["audio"]), n_mels=128).to(
            "cuda"
        )
        for timestamps in (False, True):
            options = whisper.DecodingOptions(
                language="en",
                task="transcribe",
                fp16=True,
                without_timestamps=not timestamps,
                sample_len=220,
                prompt=probe["prompt"],
            )
            start = time.perf_counter()
            with torch.inference_mode():
                decoded = whisper.decode(model, mel, options)
            results.append(
                {
                    "recordingId": probe["id"],
                    "selection": probe["selection"],
                    "timestamps": timestamps,
                    "nativeHypothesis": decoded.text,
                    "hfHypothesis": probe["hfHypothesis"],
                    "exactTextMatch": decoded.text == probe["hfHypothesis"],
                    "outputTokens": len(decoded.tokens),
                    "seconds": time.perf_counter() - start,
                    "possibleTokenLimit": len(decoded.tokens) >= 220,
                }
            )
        print(f"native decoder audit {index + 1}/{len(probes)}", flush=True)
    common.write_json(
        output,
        {
            "scriptSha256": common.sha256(Path(__file__)),
            "model": common.MODELS["mixed"],
            "dimensions": dims,
            "weightLoad": "strict; tied projection verified; no missing/unexpected keys",
            "precision": (
                "FP16 weights/activations, FP32 LayerNorm as required by OpenAI implementation"
            ),
            "purpose": "bounded decoder sensitivity check, not a new aggregate WER comparison",
            "decoding": "greedy, en, transcribe, max 220, native suppress_tokens=-1",
            "results": results,
        },
    )
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
