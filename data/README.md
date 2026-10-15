# Dados de triagem de TI

Conjunto de chamados sintéticos criado para este trabalho em 15/10/2026. Não contém chamados de clientes, nomes, senhas nem dados pessoais. Licença MIT do projeto. Os pesos Qwen possuem licença própria, Apache-2.0.

São 104 exemplos em português: 64 de treino (16 por classe), 16 de validação (4 por classe) e 24 de teste (6 por classe). As frases são escritas separadamente por partição, sem expandir o mesmo template com sufixos. `scripts/build_dataset.py` contém todas as frases e recria os arquivos JSONL. Há uma checagem de duplicação literal; isso não elimina similaridade semântica.

Campos: `id` estável, `text` (chamado), `label` (resposta esperada). Classes: ACESSO, REDE, SOFTWARE e HARDWARE. As categorias são determinadas pela causa descrita, não pela presença de um equipamento: erro em aplicativo no notebook é SOFTWARE; falha física no notebook é HARDWARE.

A validação decide a taxa de aprendizado; o teste só é executado após a escolha. O teste compara o modelo original com o adaptador selecionado. Nenhum treinamento lê validação ou teste. Os quatro exemplos do frontend não são uma métrica de desempenho.

Este corpus é pequeno, balanceado e contém casos simples de uma única categoria. Não representa a distribuição de um help desk real, não mede privacidade, segurança ou robustez a instruções maliciosas. A autoria sintética exige declarar a limitação no relatório. Para extensão: obter chamados anonimizados com autorização, rotular por duas pessoas, dividir por origem/tempo e repetir com múltiplas seeds, mantendo um novo teste intocado.
