"""Interface local: python app.py. Nao publica dados nem cria link publico."""
import threading

from src.common import ROOT, load_base, predict, read_json, seed_all
from peft import PeftModel
import gradio as gr
import torch

LOCK = threading.Lock()
MODEL = None
TOKENIZER = None


def classify(text, variant, temperature):
    global MODEL, TOKENIZER
    with LOCK:
        if not text.strip():
            raise gr.Error("Digite o chamado antes de classificar.")
        if len(text) > 1200:
            raise gr.Error("Limite de 1200 caracteres por chamado.")
        try:
            if MODEL is None:
                selection = read_json(ROOT / "reports" / "selection.json")
                MODEL, TOKENIZER = load_base()
                MODEL = PeftModel.from_pretrained(MODEL, ROOT / "adapters" / selection["selected"])
            seed_all(42)
            if variant == "Original":
                with MODEL.disable_adapter():
                    result = predict(MODEL, TOKENIZER, text, temperature)
            else:
                result = predict(MODEL, TOKENIZER, text, temperature)
            return result["response"], {"categoria": result["prediction"],
                "latencia_s": round(result["seconds"], 3), "GPU": torch.cuda.get_device_name(0),
                "dispositivo": result["device"], "modelo": variant,
                "temperatura": temperature, "VRAM_alocada_MiB": round(torch.cuda.memory_allocated() / 1024**2, 1)}
        except FileNotFoundError:
            raise gr.Error("Execute primeiro: python -m src.run_experiments") from None
        except (RuntimeError, ValueError) as error:
            raise gr.Error(str(error)) from None


with gr.Blocks(title="Triagem TI | LLM local", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Triagem TI\nUma LLM local para classificar chamados em português. Compare o modelo original com o ajuste LoRA na sua GPU.")
    with gr.Row():
        with gr.Column(scale=3):
            text = gr.Textbox(label="Descreva o chamado", placeholder="Esqueci minha senha e não consigo entrar no sistema.", lines=5, max_lines=10)
            variant = gr.Radio(["Ajustado com LoRA", "Original"], value="Ajustado com LoRA", label="Modelo")
            temperature = gr.Slider(0, 1, value=0, step=0.1, label="Temperatura (0 = resposta determinística)")
            button = gr.Button("Classificar chamado", variant="primary")
        with gr.Column(scale=2):
            response = gr.Textbox(label="Resposta da LLM", interactive=False)
            details = gr.JSON(label="Execução na GPU")
    gr.Examples([["Minha conta foi bloqueada após várias tentativas de login."],
                 ["O Wi-Fi do escritório desconecta a cada minuto."],
                 ["O editor de texto fecha ao abrir um documento."],
                 ["O monitor não liga mesmo conectado à tomada."]], inputs=text)
    gr.Markdown("Categorias: **ACESSO · REDE · SOFTWARE · HARDWARE**. Protótipo acadêmico com dados sintéticos; respostas inválidas exigem revisão humana. Chamados ambíguos ou fora do escopo não foram cobertos pelo treinamento.")
    button.click(classify, [text, variant, temperature], [response, details], api_name="classify")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860, share=False)
