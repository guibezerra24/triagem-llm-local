"""Verificacoes relevantes de dados, mascara e integracao HTTP com inferencia GPU.

Execute com o frontend ativo: python scripts/verify_project.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import LABELS, encode_training, read_json, require_gpu, rows, write_json
from src.metrics import classification_metrics
from transformers import AutoTokenizer
from gradio_client import Client
import torch


def main():
    require_gpu()
    checks = []
    splits = {split: rows(split) for split in ["train", "validation", "test"]}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not ({r["text"].casefold() for r in splits[a]} & {r["text"].casefold() for r in splits[b]})
    checks.append("Nenhuma duplicacao literal entre particoes")
    config = read_json(ROOT / "configs" / "experiments.json")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], cache_dir=ROOT / ".cache" / "hub")
    for row in splits["train"]:
        encoded = encode_training(tokenizer, row, config["max_length"])
        assert encoded["input_ids"].is_cuda and encoded["labels"].is_cuda
        supervised = encoded["labels"][0][encoded["labels"][0] != -100]
        decoded = tokenizer.decode(supervised, skip_special_tokens=True).strip()
        assert decoded == row["label"], (row["id"], decoded)
        assert (encoded["labels"] == -100).sum().item() > 0
    checks.append("64 exemplos: prompts mascarados e respostas completas em tensores CUDA")
    # Caso conhecido: erro de formato conta como falso negativo, sem desaparecer da metrica.
    fixture = [{"label": label, "prediction": label, "seconds": 1.0} for label in LABELS]
    fixture.append({"label": "ACESSO", "prediction": "INVALIDO", "seconds": 1.0})
    metrics = classification_metrics(fixture)
    assert metrics["accuracy"] == 0.8 and metrics["valid_rate"] == 0.8
    assert abs(metrics["macro_f1"] - (2 / 3 + 3) / 4) < 1e-9
    checks.append("F1, acuracia e respostas invalidas conferidos com resultado conhecido")
    client = Client("http://127.0.0.1:7860", verbose=False)
    outputs = []
    for model, temperature in [("Original", 0), ("Ajustado com LoRA", 0), ("Ajustado com LoRA", 0.7)]:
        response, details = client.predict("Preciso trocar a senha de acesso da minha conta.", model, temperature, api_name="/classify")
        assert response and details["dispositivo"] == "cuda:0"
        assert details["modelo"] == model and details["temperatura"] == temperature
        outputs.append({"response": response, **details})
    checks.append("Frontend HTTP: original, LoRA deterministico e LoRA com amostragem executados em CUDA")
    try:
        client.predict("", "Original", 0, api_name="/classify")
    except Exception as error:
        assert "Digite o chamado" in str(error), str(error)
    else:
        raise AssertionError("Prompt vazio deveria ser rejeitado")
    checks.append("Frontend rejeita prompt vazio")
    expected = read_json(ROOT / "reports" / "manifest.json")["fingerprint"]
    current = hashlib.sha256((ROOT / "configs" / "experiments.json").read_bytes() + b"".join((ROOT / "data" / f"{s}.jsonl").read_bytes() for s in ["train", "validation", "test"]) + b"".join(p.read_bytes() for p in sorted((ROOT / "src").glob("*.py")))).hexdigest()
    assert current == expected, "Codigo/configuracao/dados mudaram desde o treinamento"
    checks.append("Manifesto corresponde ao codigo, configuracao e dados entregues")
    write_json(ROOT / "reports" / "verification.json", {"passed": True, "gpu": torch.cuda.get_device_name(0), "checks": checks, "frontend_outputs": outputs})
    print(json.dumps({"passed": True, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
