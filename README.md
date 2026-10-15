# Triagem TI — ajuste de uma LLM local na GTX 1650

Trabalho acadêmico de **Guilherme Rezende (RM 98508), Matheus Brisqui (RM 97892) e Gustavo Brisqui (RM 97969)**. A aplicação recebe um chamado em português e gera uma categoria: **ACESSO, REDE, SOFTWARE ou HARDWARE**. O objetivo é comparar uma LLM original com dois ajustes de treinamento e demonstrar inferência local na GPU.

## Escolha do modelo e do problema

Modelo: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), aproximadamente 0,49 bilhão de parâmetros, licença Apache-2.0. A revisão exata do modelo fica congelada em `configs/experiments.json` na primeira execução. É um modelo multilíngue de pequeno porte adequado a uma tarefa delimitada com saída curta. Português é avaliado neste corpus, sem presumir desempenho geral a partir do suporte multilíngue.

Hardware detectado: **NVIDIA GeForce GTX 1650, 4096 MiB**, driver 595.97. A escolha de 0.5B e LoRA prioriza caber na placa existente. Um modelo de 7B precisaria de técnicas e memória adicionais. O projeto faz **fine-tuning supervisionado (SFT)** de um modelo pré-treinado; não é pré-treinamento de uma LLM do zero. Uma classificação tradicional também poderia resolver esta tarefa; a LLM foi escolhida para atender à proposta acadêmica e estudar adaptação por instrução.

O ajuste LoRA treina matrizes pequenas nas projeções `q_proj` e `v_proj`, preservando os pesos originais. Usamos FP32, microbatch de 1, acumulação de gradientes e checkpointing de ativações. Não dependemos de bitsandbytes ou Flash Attention no Windows. O frontend e todas as execuções do modelo exigem CUDA; o código interrompe se a GPU não estiver disponível. Preparação de texto, cálculo de métricas e interface usam CPU, como é normal; treinamento, geração e o teste de operações tensoriais usam GPU.

## Instalação e CUDA

Ambiente validado nesta entrega: Windows e Python 3.13. Consulte `reports/gpu.json` e `requirements-lock.txt` para as versões efetivamente executadas. Reserve espaço para o PyTorch CUDA e o download inicial do modelo. Há acesso à internet somente para instalar pacotes e baixar pesos; após o cache estar completo, o modelo pode funcionar localmente.

Abra esta pasta no VSCode e execute no PowerShell:

```powershell
pwsh -File scripts/setup.ps1
.\.venv\Scripts\python.exe -m src.run_experiments
.\.venv\Scripts\python.exe app.py
```

Se só houver Windows PowerShell, use `powershell -File scripts/setup.ps1`. Se a política local bloquear scripts, execute os comandos do arquivo manualmente no terminal; não é necessário mudar a política do sistema.

O instalador usa o pacote oficial **PyTorch 2.7.1 + CUDA 12.6**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.gpu_check
```

O valor **CUDA 13.2 no `nvidia-smi` é a capacidade do driver**, não a versão do runtime usada pelo PyTorch. `torch.version.cuda` deve indicar **12.6**, e `torch.cuda.is_available()` deve retornar `True`. O driver atual executou o runtime 12.6 neste equipamento. Os pacotes oficiais incluem bibliotecas de runtime; o CUDA Toolkit completo e o `nvcc` só são necessários se for compilar extensões CUDA, o que este projeto não faz. A instalação não troca nem rebaixa o driver da máquina.

Referências: [instalações oficiais do PyTorch](https://pytorch.org/get-started/previous-versions/), [compatibilidade NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [LoRA no PEFT](https://huggingface.co/docs/peft/en/package_reference/lora).

## Dados e desenho experimental

Corpus sintético autoral de **104 chamados**: 64 de treino, 16 de validação e 24 de teste. Todas as partições são balanceadas nas quatro classes. A documentação de origem, critérios e limitações está em [`data/README.md`](data/README.md). As frases estão versionadas em JSONL. Não há dados pessoais ou credenciais reais.

| Configuração | Taxa de aprendizado | Rank / alpha | Dropout | Épocas | Microbatch / acumulação |
|---|---:|---:|---:|---:|---:|
| Original | sem ajuste | — | — | 0 | — |
| lora_lr1e4 | 0,0001 | 8 / 16 | 0,05 | 2 | 1 / 4 |
| lora_lr3e4 | 0,0003 | 8 / 16 | 0,05 | 2 | 1 / 4 |

O experimento altera **uma variável: taxa de aprendizado**. Os demais parâmetros ficam constantes. O batch efetivo é 4; há 16 atualizações por época e 32 por experimento. Outras definições: AdamW, weight decay 0,01, clipping 1,0, seed 42, comprimento máximo 192 tokens, apenas tokens da resposta entram na loss. Textos que ultrapassam o limite interrompem o treinamento em vez de truncar a resposta.

A geração da avaliação é livre, sem restringir tokens a uma lista: `do_sample=False`, no máximo 12 novos tokens. A validação escolhe o maior **F1 macro**; em empate, acurácia e menor taxa de aprendizado. A escolha é salva em `reports/selection.json` **antes** do teste. Somente o modelo original e o adaptador escolhido são avaliados no teste. O conjunto de teste não decide hiperparâmetros.

Métricas: acurácia, precisão/recall/F1 por classe, F1 macro, taxa de formato válido, matriz de confusão e latência por chamado. O parser aceita apenas uma categoria exata após remover espaços externos e converter para maiúsculas. Prosa adicional e múltiplas categorias contam como `INVALIDO`, sem tentar adivinhar a resposta. A latência mede a geração com sincronização CUDA e exclui a tokenização, sem warmup separado; não é um benchmark estabilizado de produção.

## Execução e resultados

```powershell
# Diagnóstico real: multiplicação de matrizes + backward na GPU
.\.venv\Scripts\python.exe -m src.gpu_check

# Baseline, dois treinamentos, validação e teste final
.\.venv\Scripts\python.exe -m src.run_experiments

# Retomar apenas etapas concluídas com mesmo código/configuração/dados
.\.venv\Scripts\python.exe -m src.run_experiments --resume

# Gerar relatório a partir dos resultados reais
.\.venv\Scripts\python.exe scripts/build_report.py

# Interface local
.\.venv\Scripts\python.exe app.py
```

A interface fica em **http://127.0.0.1:7860**, recebe texto livre, permite trocar entre modelo original e ajustado e alterar temperatura. Temperatura 0 usa geração determinística; acima de zero ativa amostragem com top-p 0,9. Essa exploração na interface não altera os resultados determinísticos do protocolo. O adaptador é carregado uma vez e pode ser desabilitado para comparar o modelo original, evitando duas cópias da LLM na VRAM. Nenhuma API de LLM paga é utilizada.

Para verificar máscara da loss, separação dos dados, métricas e integração HTTP com os dois modos de modelo na GPU, mantenha o frontend ativo e execute `.\.venv\Scripts\python.exe scripts/verify_project.py`. A execução entregue passou nas verificações e está registrada em `reports/verification.json`.

O **relatório completo está em [`reports/RESULTADOS.md`](reports/RESULTADOS.md)**. As respostas individuais ficam em `reports/*_validation.json` e `reports/*_test.json`; as curvas de loss em `reports/*_history.json`; tempo, VRAM e quantidade de parâmetros em `reports/*_training.json`. `reports/manifest.json` registra hashes dos dados e do código/configuração. `reports/summary.json` consolida os resultados. Os adaptadores treinados ficam em `adapters/`; os pesos originais são baixados do Hugging Face e não entram no Git.

No teste de 24 chamados, o modelo original acertou **11 (45,83%)**, e o ajuste selecionado acertou **20 (83,33%)**. São apenas seis exemplos de teste por classe e uma seed; relatos reais, ambiguidade, múltiplas categorias e mudanças de linguagem exigem avaliação adicional. O protótipo não executa ações administrativas e requer revisão humana antes de qualquer uso operacional.

## Evidência no VSCode

`reports/gpu.txt` registra saída real do `nvidia-smi`, versão do PyTorch/runtime, nome da placa, dispositivo `cuda:0` e gradientes finitos. Abra esse relatório e `src/gpu_check.py` no VSCode. A captura entregue em `reports/gpu_vscode.png` mostra a execução do diagnóstico no terminal integrado do VSCode.

Para reproduzir em sala: selecione o interpretador `.venv\Scripts\python.exe`, abra Terminal → Executar tarefa e escolha **GPU: verificar CUDA e backward**. O arquivo `.vscode/tasks.json` também contém tarefas para treinar e abrir o frontend.

As extensões **GPU Environment** e **GPU Monitor**, sugeridas no enunciado, podem ser pesquisadas no Marketplace do VSCode. Verifique editor, suporte à sua plataforma e requisitos antes de instalar; não são necessárias para o CUDA funcionar. Não afirmamos que estão instaladas. O diagnóstico PyTorch mais `nvidia-smi` comprova uso computacional além da presença física da placa.

## Integrantes

| Integrante | RM |
|---|---|
| Guilherme Rezende | 98508 |
| Matheus Brisqui | 97892 |
| Gustavo Brisqui | 97969 |

**Entrega em trio autorizada pelo professor, conforme informado pelo grupo.** O enunciado original previa quatro a cinco integrantes. Os dados dos participantes também estão em `team.json`.

Roteiro de apresentação (8–10 minutos): problema e categorias (1 min); escolha do modelo/LoRA (1 min); GPU/CUDA e print (1 min); dados e controle de vazamento (1 min); parâmetros e comparação (2 min); demo com um chamado novo (1–2 min); limitações, reprodutibilidade e GitHub (1 min).

## Estrutura e GitHub

```text
app.py                  frontend Gradio com inferência CUDA
src/common.py           modelo, prompt, máscara de loss e geração
src/gpu_check.py        diagnóstico GPU com backward real
src/run_experiments.py  treinamento, seleção e avaliação
src/metrics.py          métricas e matriz de confusão
configs/                hiperparâmetros e revisão do modelo
data/                   treino, validação, teste e origem
adapters/               pesos LoRA e configuração
reports/                resultados, respostas, logs e capturas reais
scripts/                instalação, dados e relatório
.vscode/                tarefas e configuração do editor
```

O código e os dados sintéticos são MIT; os pesos do modelo base seguem Apache-2.0. Os adaptadores devem ser usados junto do modelo e de sua licença. Não versionar `.venv`, cache do modelo, tokens ou arquivos `.env`. A aplicação abre somente no endereço local, sem link público do Gradio.

Repositório: [guibezerra24/triagem-llm-local](https://github.com/guibezerra24/triagem-llm-local). Os adaptadores LoRA são pequenos e podem ser versionados; as cópias redundantes do tokenizer geradas no treinamento são ignoradas, pois o carregamento usa a revisão fixa do modelo base. Para publicar manualmente em um repositório vazio:

```powershell
git init -b main
git add .
git commit -m "Projeto de fine-tuning local com LoRA, GPU e frontend"
git remote add origin https://github.com/guibezerra24/triagem-llm-local.git
git push -u origin main
```

## Conferência com o enunciado

| Exigência | Entrega |
|---|---|
| Escolher LLM | Qwen2.5-0.5B-Instruct e revisão congelada |
| Problema específico | Triagem de chamados de TI em quatro categorias |
| Diferentes parâmetros | Duas taxas de aprendizado mantendo demais fatores fixos |
| Avaliar respostas | Baseline, validação, teste separado, respostas brutas e métricas |
| Todos os testes de modelo na GPU | CUDA obrigatório em treino e inferência, com evidência de dispositivo |
| Frontend para criar prompt | Gradio local, texto livre, modelo e temperatura |
| Print da GPU no VSCode | `reports/gpu_vscode.png` e diagnóstico real |
| Documentação e código no GitHub | Projeto organizado para o repositório e publicação verificável |
| Grupo | Três integrantes identificados; entrega em trio autorizada pelo professor, conforme informado pelo grupo |

A revisão detalhada dos requisitos e seus arquivos de comprovação está em [`reports/REQUISITOS.md`](reports/REQUISITOS.md).
