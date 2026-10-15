"""Gera tabelas e figuras apenas de arquivos efetivamente produzidos na GPU."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["ACESSO", "REDE", "SOFTWARE", "HARDWARE"]


def read(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows) + "\n"


def main():
    summary = read("summary.json")
    gpu = read("gpu.json")
    selected = summary["selected"]
    original = summary["test"]["baseline"]
    adjusted = summary["test"][selected]
    exp = next(r for r in summary["experiments"] if r["name"] == selected)
    team = json.loads((ROOT / "team.json").read_text(encoding="utf-8"))
    participants = "; ".join(f"{person['name']} (RM {person['rm']})" for person in team["participants"])
    lines = ["# Resultados reais — Triagem TI\n",
        f"**Integrantes:** {participants}.\n",
        team["professor_authorization"] + "\n",
        f"Execução local na **{gpu['gpu']}**, PyTorch **{gpu['torch']}**, runtime CUDA **{gpu['cuda_runtime']}**. Operações e backward confirmados em `{gpu['operation_device']}`. Os números abaixo são lidos dos JSONs desta entrega, sem valores simulados.\n",
        "## Seleção pela validação (16 exemplos)\n"]
    validation_rows = [["Original", f"{summary['baseline_validation']['accuracy']:.2%}", f"{summary['baseline_validation']['macro_f1']:.4f}", "—", "—"]]
    for result in summary["experiments"]:
        validation_rows.append([result["name"], f"{result['validation']['accuracy']:.2%}", f"{result['validation']['macro_f1']:.4f}",
            f"{result['training']['seconds']:.2f}", f"{result['training']['peak_allocated_mib']:.1f}"])
    lines += [table(["Modelo", "Acurácia", "F1 macro", "Treino (s)", "Pico alocado (MiB)"], validation_rows),
        f"Selecionado: **{selected}**, pela validação. A taxa de aprendizado maior não deve ser escolhida só por reduzir a loss de treino: observe as confusões por classe.\n",
        "Tempos de treinamento são observações de uma execução sequencial, sem aquecimento controlado. A diferença de tempo não demonstra que uma taxa de aprendizado é mais rápida. O pico alocado pelo PyTorch não inclui toda a VRAM usada pelo Windows e outros aplicativos.\n",
        "## Teste final intocado (24 exemplos, 6 por classe)\n",
        table(["Modelo", "Acertos", "Acurácia", "F1 macro", "Formato válido", "Latência média (s)"],
            [[name, f"{round(metrics['accuracy'] * metrics['n'])}/{metrics['n']}", f"{metrics['accuracy']:.2%}", f"{metrics['macro_f1']:.4f}",
              f"{metrics['valid_rate']:.2%}", f"{metrics['mean_seconds']:.3f}"] for name, metrics in summary["test"].items()]),
        f"Variação de acurácia neste conjunto: **{(adjusted['accuracy'] - original['accuracy']) * 100:+.2f} pontos percentuais**. F1 macro: **{original['macro_f1']:.4f} → {adjusted['macro_f1']:.4f}**. Não foi feito teste de significância estatística.\n",
        "![Comparação de acurácia e F1 macro](comparacao.png)\n",
        "## Métricas por classe do ajuste escolhido\n",
        table(["Classe", "Precisão", "Recall", "F1", "Suporte"],
            [[label, f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}", m["support"]] for label, m in adjusted["per_class"].items()]),
        "![Matriz de confusão do modelo ajustado](confusao.png)\n",
        "## Respostas do teste: antes e depois\n"]
    before = {r["id"]: r for r in read("baseline_test.json")["records"]}
    after = read(f"{selected}_test.json")["records"]
    comparison = [[r["text"], r["label"], before[r["id"]]["response"], r["response"], "sim" if r["label"] == r["prediction"] else "não"] for r in after]
    lines += [table(["Chamado", "Esperado", "Original", "LoRA", "Acertou após ajuste?"], comparison),
        "## Erros que permanecem\n"]
    for r in after:
        if r["label"] != r["prediction"]:
            lines.append(f"- {r['id']}: “{r['text']}” — esperado **{r['label']}**, gerado **{r['response']}**.\n")
    lines += ["\nAs saídas brutas, IDs, dispositivo e latência de cada exemplo estão nos JSONs; não houve correção manual das respostas.\n",
        "## Treinamento e recursos\n",
        f"- Parâmetros treináveis: **{exp['training']['trainable_parameters']:,}**, de **{exp['training']['all_parameters']:,}** incluindo LoRA (**{100 * exp['training']['trainable_parameters'] / exp['training']['all_parameters']:.3f}%**).\n",
        f"- Atualizações do otimizador por experimento: **{exp['training']['optimizer_steps']}**. Dispositivo **{exp['training']['device']}**, precisão **{exp['training']['dtype']}**.\n",
        "- A loss supervisiona somente tokens da resposta e do encerramento da conversa. Os prompts são mascarados com -100.\n",
        "\n![Loss durante os dois treinamentos](loss.png)\n",
        "## Evidências visuais\n\n![GPU e diagnóstico dentro do VSCode](gpu_vscode.png)\n\n![Frontend local com inferência na GPU](frontend.jpg)\n",
        "## Limitações e conclusão\n",
        "O ajuste melhorou as métricas observadas neste teste sintético pequeno. Isso não estabelece desempenho em chamados reais. Uma seed, frases curtas e classes balanceadas não permitem estimar a robustez em produção. Categorias ambíguas, múltiplos problemas, dados fora do escopo e ataques por prompt não foram avaliados sistematicamente. A loss menor do segundo ajuste não garantiu melhor F1 de validação.\n",
        "A latência mede geração sincronizada com CUDA, sem warmup separado; o primeiro exemplo pode incluir inicialização. Não inclui download, carregamento dos pesos nem tempo de transporte do frontend. O adaptador adiciona operações ao modelo, e a comparação de velocidade exige repetições controladas.\n",
        "Para uma extensão do trabalho: coletar chamados autorizados e anonimizados, separar por tempo/origem, usar dupla rotulagem, ampliar exemplos por classe, treinar com três ou mais seeds e reservar um novo teste antes de explorar outras configurações.\n"]
    (ROOT / "reports" / "RESULTADOS.md").write_text("\n".join(lines), encoding="utf-8")
    with (ROOT / "reports" / "respostas_teste.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["chamado", "esperado", "original", "lora", "acertou_lora"])
        writer.writerows(comparison)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    x = np.arange(2)
    for offset, key, label, color in [(-0.18, "accuracy", "Acurácia", "#2655b8"), (0.18, "macro_f1", "F1 macro", "#008778")]:
        vals = [original[key], adjusted[key]]
        bars = ax.bar(x + offset, vals, width=0.34, label=label, color=color)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=4)
    ax.set(xticks=x, xticklabels=["Original", "LoRA selecionado"], ylim=(0, 1.13), ylabel="Escore", title="Teste final · 24 chamados sintéticos · GTX 1650")
    ax.legend(loc="upper left")
    fig.savefig(ROOT / "reports" / "comparacao.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for result in summary["experiments"]:
        history = read(f"{result['name']}_history.json")
        ax.plot([r["step"] for r in history], [r["loss"] for r in history], label=result["name"])
    ax.set(xlabel="Atualização do otimizador", ylabel="Loss média do grupo", title="Loss de treino · seed 42 · 2 épocas")
    ax.legend()
    fig.savefig(ROOT / "reports" / "loss.png", dpi=160)
    plt.close(fig)
    columns = LABELS + ["INVALIDO"]
    matrix = np.array([[adjusted["confusion_matrix"][label][pred] for pred in columns] for label in LABELS])
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=6)
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, str(value), ha="center", va="center", color="white" if value >= 4 else "black")
    ax.set(xticks=np.arange(len(columns)), xticklabels=columns, yticks=np.arange(4), yticklabels=LABELS,
           xlabel="Categoria gerada", ylabel="Categoria esperada", title=f"Matriz de confusão · {selected} · teste")
    fig.savefig(ROOT / "reports" / "confusao.png", dpi=160)
    plt.close(fig)
    print("Relatorio, CSV e tres figuras gerados a partir de resultados reais.")


if __name__ == "__main__":
    main()
