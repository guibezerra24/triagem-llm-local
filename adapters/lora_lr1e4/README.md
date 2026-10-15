---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
license: apache-2.0
language:
- pt
tags:
- lora
- text-generation
---

# Adaptador LoRA para triagem de TI

Ajuste supervisionado em 64 chamados sinteticos em portugues. Classes: ACESSO, REDE, SOFTWARE, HARDWARE. Modelo base e tokenizer: Qwen2.5-0.5B-Instruct, revisao 7ae557604adf67be50417f59c2c2f167def9a775.

Configuracao: lora_lr1e4. F1 macro de validacao: 0.8730.

Nao comprovado em chamados reais. Consulte README.md e reports/RESULTADOS.md na raiz para protocolo, resultados e limitacoes.
