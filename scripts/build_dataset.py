"""Corpus sintetico autoral: cada split possui frases distintas, sem expansao por template."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = {
    "train": {
        "ACESSO": [
            "Esqueci a senha do portal da empresa.", "Meu login foi bloqueado depois de errar a senha.",
            "Preciso de permissao para acessar a pasta financeira.", "Minha conta corporativa foi desativada.",
            "O codigo de autenticacao em duas etapas nao chega.", "Quero redefinir a senha do email.",
            "Nao tenho autorizacao para abrir o painel administrativo.", "Troquei de celular e perdi o autenticador da conta.",
            "Recebo acesso negado ao entrar com meu usuario.", "Meu perfil nao permite editar o cadastro.",
            "O sistema diz que minha senha expirou.", "Preciso criar uma conta para um novo funcionario.",
            "Quero remover o acesso de um colaborador desligado.", "Nao consigo desbloquear meu usuario no dominio.",
            "Solicito liberacao do meu perfil para consultar relatorios.", "Meu token de verificacao de identidade venceu."
        ],
        "REDE": [
            "A internet caiu em todo o escritorio.", "O Wi-Fi nao conecta no notebook.",
            "A conexao de rede fica oscilando durante as reunioes.", "Nenhum site abre na rede da empresa.",
            "O cabo de rede esta conectado mas nao tenho internet.", "A VPN desconecta a cada cinco minutos.",
            "O computador nao recebe endereco IP do roteador.", "O sinal do Wi-Fi esta muito fraco na sala.",
            "O DNS da rede nao resolve os enderecos dos sites.", "A internet ficou lenta em todas as maquinas.",
            "A conexao por Ethernet aparece como limitada.", "O roteador nao distribui internet para os usuarios.",
            "Tenho perda de pacotes na conexao corporativa.", "O ponto de acesso sem fio desapareceu da lista.",
            "A rede interna nao alcanca o servidor por IP.", "Depois da troca do modem ficamos sem conexao."
        ],
        "SOFTWARE": [
            "O editor de texto fecha sozinho ao abrir arquivos.", "Preciso instalar o aplicativo de videoconferencia.",
            "A planilha mostra erro e o programa trava.", "A atualizacao do navegador falhou.",
            "O sistema de estoque apresenta uma excecao na tela.", "O aplicativo de email nao inicia depois da atualizacao.",
            "Quero desinstalar uma versao antiga do editor de imagens.", "O leitor de PDF encerra ao abrir um documento.",
            "O instalador do programa para com um erro.", "O aplicativo fica congelado ao salvar o projeto.",
            "A nova versao do sistema operacional nao conclui a atualizacao.", "A ferramenta de desenho nao exporta os arquivos.",
            "O software de reuniao abre uma janela em branco.", "O programa de notas duplica os registros ao salvar.",
            "A extensao do navegador provoca erro na pagina.", "O aplicativo de gestao apresenta falha de execucao."
        ],
        "HARDWARE": [
            "O teclado parou de reconhecer varias teclas.", "O monitor nao liga mesmo conectado na tomada.",
            "O mouse esta com o botao fisico quebrado.", "O computador nao liga e a fonte nao acende.",
            "O ventilador do notebook faz muito barulho.", "A bateria do notebook nao segura carga.",
            "O disco rigido faz estalos e nao e detectado na BIOS.", "A tela do notebook esta quebrada.",
            "A porta USB esta fisicamente solta.", "O carregador apresenta fio partido.",
            "A impressora esta com papel preso no mecanismo.", "O computador superaquece e desliga.",
            "O pente de memoria RAM apresenta defeito no teste fisico.", "O cabo de energia do monitor esta danificado.",
            "O microfone tem a haste quebrada.", "A dobradica do notebook se soltou da carcaca."
        ],
    },
    "validation": {
        "ACESSO": ["Meu usuario exige troca de credencial para entrar.", "Preciso habilitar meu perfil no painel de vendas.",
                   "A verificacao em duas etapas rejeita meu codigo.", "Solicito recuperacao da minha conta de trabalho."],
        "REDE": ["Os computadores do andar perderam conexao com a internet.", "A rede sem fio cai quando entro na sala de reuniao.",
                 "O servidor DNS corporativo esta inacessivel pela rede.", "O tunel VPN interrompe a conexao repetidamente."],
        "SOFTWARE": ["O aplicativo de apresentacoes fecha quando adiciono um slide.", "A instalacao do editor de video nao termina.",
                     "O navegador exibe uma falha interna ao iniciar.", "O programa de desenho congela ao importar uma imagem."],
        "HARDWARE": ["Derrubei agua no teclado e as teclas nao funcionam.", "A fonte do desktop queimou apos uma queda de energia.",
                     "O cooler da maquina esta travado.", "A bateria do computador portatil esta estufada."],
    },
    "test": {
        "ACESSO": ["Nao lembro mais a credencial usada para entrar no RH.", "Um colega novo precisa de um usuario no sistema.",
                   "Meu perfil permite visualizar, mas preciso de autorizacao para alterar dados.",
                   "Perdi o dispositivo que gera os codigos de login.", "O portal recusa a entrada porque a conta venceu.",
                   "Solicito revogar as permissoes de uma conta antiga."],
        "REDE": ["Toda a equipe ficou sem acesso a sites ao mesmo tempo.", "O notebook se conecta ao ponto sem fio mas fica sem internet.",
                 "A ligacao com a rede da matriz cai de tempos em tempos.", "Ha muita perda de pacotes entre os dois predios.",
                 "Nenhuma maquina recebe IP automaticamente nesta sub-rede.", "O sinal wireless nao chega na recepcao."],
        "SOFTWARE": ["O aplicativo de contabilidade encerra quando gero o balanco.", "A atualizacao do programa de edicao retorna um erro.",
                     "Ao clicar em salvar no editor, o documento desaparece e o aplicativo trava.",
                     "O visualizador de imagens apresenta uma excecao ao carregar fotos.",
                     "Preciso instalar o utilitario de compactacao de arquivos.", "A ferramenta de calendario abre mas fica sem responder."],
        "HARDWARE": ["A tecla de espaco se desprendeu do teclado.", "O visor sofreu uma trinca e ficou com manchas.",
                     "A ventoinha nao gira e o gabinete esquenta muito.", "O conector do carregador quebrou dentro do equipamento.",
                     "O botao esquerdo do mouse afundou e nao retorna.", "O mecanismo da impressora nao puxa as folhas da bandeja."],
    },
}


def main():
    output = ROOT / "data"
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split, classes in DATA.items():
        records = []
        for label, texts in classes.items():
            for i, text in enumerate(texts, 1):
                assert text.casefold() not in seen, "Frase duplicada entre splits"
                seen.add(text.casefold())
                records.append({"id": f"{split}_{label.lower()}_{i:02d}", "text": text, "label": label})
        (output / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
        print(split, len(records), dict(Counter(r["label"] for r in records)))


if __name__ == "__main__":
    main()
