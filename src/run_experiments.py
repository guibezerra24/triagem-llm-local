"""Baseline, dois ajustes LoRA, selecao por validacao e teste final GPU."""
import argparse
import gc
import hashlib
import time

import torch
from huggingface_hub import model_info
from peft import LoraConfig, PeftModel, get_peft_model
from .common import ROOT, encode_training, load_base, predict, read_json, rows, seed_all, write_json
from .metrics import classification_metrics


def evaluate(model, tokenizer, split, name):
    records = []
    for i, row in enumerate(rows(split), 1):
        result = predict(model, tokenizer, row["text"])
        records.append({**row, **result})
        print(f"{name} {split} {i}/{len(rows(split))}: {row['label']} -> {result['prediction']}", flush=True)
    metrics = classification_metrics(records)
    write_json(ROOT / "reports" / f"{name}_{split}.json", {"metrics": metrics, "records": records})
    return metrics


def train(exp, config):
    seed_all(config["seed"])
    model, tokenizer = load_base()
    model = get_peft_model(model, LoraConfig(r=exp["rank"], lora_alpha=exp["alpha"],
        lora_dropout=exp["dropout"], target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    data = [encode_training(tokenizer, row, config["max_length"]) for row in rows("train")]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=exp["learning_rate"], weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    history = []
    steps = 0
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(data)).tolist()
        for begin in range(0, len(order), config["gradient_accumulation"]):
            group = order[begin:begin + config["gradient_accumulation"]]
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for index in group:
                loss = model(**data[index]).loss
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Loss nao finita: experimento interrompido.")
                loss_sum += loss.item()
                (loss / len(group)).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm).item():
                raise RuntimeError("Gradiente nao finito: experimento interrompido.")
            optimizer.step()
            steps += 1
            item = {"epoch": epoch + 1, "step": steps, "loss": loss_sum / len(group), "grad_norm": norm.item()}
            history.append(item)
            print(f"{exp['name']} epoch={epoch+1} step={steps} loss={item['loss']:.4f}", flush=True)
            write_json(ROOT / "reports" / f"{exp['name']}_history.json", history)
    torch.cuda.synchronize()
    training = {"seconds": time.perf_counter() - started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "all_parameters": sum(p.numel() for p in model.parameters()), "optimizer_steps": steps,
                "device": str(next(model.parameters()).device), "dtype": str(next(model.parameters()).dtype)}
    adapter = ROOT / "adapters" / exp["name"]
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    # Liberar estado do otimizador e tensores antes da inferencia.
    del optimizer, data
    gc.collect()
    torch.cuda.empty_cache()
    validation = evaluate(model, tokenizer, "validation", exp["name"])
    write_json(ROOT / "reports" / f"{exp['name']}_training.json", training)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return {"name": exp["name"], "hyperparameters": exp, "training": training, "validation": validation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Reutiliza etapas concluidas com o mesmo manifesto")
    args = parser.parse_args()
    config_path = ROOT / "configs" / "experiments.json"
    config = read_json(config_path)
    if not config.get("revision"):
        config["revision"] = model_info(config["model_id"]).sha
        write_json(config_path, config)
    fingerprint = hashlib.sha256(config_path.read_bytes() + b"".join((ROOT / "data" / f"{s}.jsonl").read_bytes() for s in ["train", "validation", "test"]) + b"".join(p.read_bytes() for p in sorted((ROOT / "src").glob("*.py")))).hexdigest()
    manifest_path = ROOT / "reports" / "manifest.json"
    if args.resume and manifest_path.exists() and read_json(manifest_path)["fingerprint"] != fingerprint:
        raise RuntimeError("Configuracao, dados ou codigo mudaram. Rode sem --resume.")
    write_json(manifest_path, {"fingerprint": fingerprint, "config": config,
        "data_sha256": {s: hashlib.sha256((ROOT / "data" / f"{s}.jsonl").read_bytes()).hexdigest() for s in ["train", "validation", "test"]}})
    seed_all(config["seed"])
    baseline_path = ROOT / "reports" / "baseline_validation.json"
    if args.resume and baseline_path.exists():
        baseline_validation = read_json(baseline_path)["metrics"]
    else:
        model, tokenizer = load_base()
        baseline_validation = evaluate(model, tokenizer, "validation", "baseline")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    results = []
    for exp in config["experiments"]:
        path = ROOT / "reports" / f"{exp['name']}_result.json"
        if args.resume and path.exists():
            result = read_json(path)
        else:
            result = train(exp, config)
            write_json(path, result)
        results.append(result)
    # Escolha congelada ANTES de ler os resultados de teste. Empate: menor LR.
    best = max(results, key=lambda r: (r["validation"]["macro_f1"], r["validation"]["accuracy"], -r["hyperparameters"]["learning_rate"]))
    write_json(ROOT / "reports" / "selection.json", {"selected": best["name"], "criterion": "validation macro_f1, accuracy; tie: lower learning rate"})
    tests = {}
    for name in ["baseline", best["name"]]:
        path = ROOT / "reports" / f"{name}_test.json"
        if args.resume and path.exists():
            tests[name] = read_json(path)["metrics"]
            continue
        model, tokenizer = load_base()
        if name != "baseline":
            model = PeftModel.from_pretrained(model, ROOT / "adapters" / name)
        tests[name] = evaluate(model, tokenizer, "test", name)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    write_json(ROOT / "reports" / "summary.json", {"baseline_validation": baseline_validation,
        "experiments": results, "selected": best["name"], "test": tests,
        "limitations": ["Dados sinteticos pequenos", "Uma seed", "Nao comprova generalizacao para chamados reais", "Latencia inclui primeira geracao sem warmup separado"]})
    print("Concluido: reports/summary.json", flush=True)


if __name__ == "__main__":
    main()
