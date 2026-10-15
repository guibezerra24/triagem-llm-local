# Revisão dos requisitos

Revisão de 15/10/2026, baseada nos arquivos e registros desta entrega. O treinamento e as métricas originais foram preservados.

## Integrantes

| Nome | RM |
|---|---|
| Guilherme Rezende | 98508 |
| Matheus Brisqui | 97892 |
| Gustavo Brisqui | 97969 |

Entrega em trio autorizada pelo professor, conforme informado pelo grupo.

## Conferência

| Requisito | Situação | Verificação | Evidência |
|---|---|---|---|
| Escolher e treinar uma LLM local | Atendido | Qwen2.5-0.5B-Instruct, revisão fixa, dois adaptadores LoRA e histórico de 32 atualizações por configuração. | configs/experiments.json; adapters/; reports/*_training.json; reports/*_history.json |
| Escolher um problema específico | Atendido | Triagem de chamados em português: ACESSO, REDE, SOFTWARE e HARDWARE. | README.md; src/common.py; data/README.md |
| Testar diferentes parâmetros | Atendido | Taxas de aprendizado 0,0001 e 0,0003, mantendo os demais fatores constantes. A integração também registra temperatura 0 e 0,7. | configs/experiments.json; reports/*_result.json; reports/verification.json |
| Verificar e avaliar as respostas | Atendido | Validação separada, comparação original/LoRA no teste, respostas brutas, acurácia, F1 macro, métricas por classe e matriz de confusão. | reports/RESULTADOS.md; reports/*_validation.json; reports/*_test.json |
| Executar os testes do modelo na GPU | Atendido | Treinamento e registros de geração em cuda:0; teste tensorial com backward finito. Dados, métricas e interface são processados em CPU. | reports/gpu.json; reports/*_training.json; reports/*_test.json; reports/verification.json |
| Frontend para inserir/criar o prompt | Atendido | Campo de texto livre, seleção do modelo, temperatura e retorno da categoria com informações da GPU. Integração HTTP aprovada. | app.py; reports/frontend.jpg; reports/verification.json |
| Print da GPU instalada e funcionando no VSCode | Atendido | Captura original do VSCode exibindo GTX 1650, runtime CUDA 12.6, cuda_available=True, cuda:0, backward_finite=True e nvidia-smi. | reports/gpu_vscode.jpg; reports/gpu.txt; src/gpu_check.py |
| Documentação e código no GitHub | Atendido | Repositório público com código, instruções, dados, configurações, adaptadores, métricas, capturas e licenças; atualização dos integrantes incluída nesta revisão. | https://github.com/guibezerra24/triagem-llm-local |
| Composição do grupo | Atendido com autorização informada | Três integrantes identificados. O grupo informou que o professor autorizou a entrega em trio, em exceção ao tamanho original de 4 a 5 pessoas. | team.json; README.md |

## Resultado conferido

- Original: 11/24 acertos (45.83%), F1 macro 0.4128.
- Ajuste selecionado: 20/24 acertos (83.33%), F1 macro 0.8293.
- Todos os registros de treinamento e geração revisados usam `cuda:0`.
- Código, configuração e dados correspondem ao hash do manifesto de treinamento.
- Os arquivos de imagem das capturas e dos gráficos estão presentes.

## Observações do enunciado

As extensões GPU Environment e GPU Monitor são sugestões de monitoramento. Elas não foram instaladas nem declaradas instaladas. O diagnóstico PyTorch e o nvidia-smi são as evidências do funcionamento da GPU.

O runtime CUDA 12.6 fornecido pelo PyTorch foi validado na GTX 1650. O CUDA Toolkit completo e o nvcc não são necessários para esta implementação. O valor 13.2 do nvidia-smi indica a compatibilidade do driver.

Os requisitos técnicos possuem evidências nos arquivos indicados. A composição em trio depende da autorização informada pelo grupo. A apresentação oral e o envio do link no ambiente da disciplina devem ser realizados pelo grupo.

## Limites da revisão

Os dados são sintéticos, com 24 exemplos de teste e uma única seed. A melhora observada não comprova generalização para chamados reais. Esta revisão verifica a consistência dos registros existentes e não cria novas medições de treinamento.
