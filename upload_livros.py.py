import os
from typing import Optional
from supabase import create_client
from openai import OpenAI
from pypdf import PdfReader  # Biblioteca para ler PDFs reais
from dotenv import load_dotenv

load_dotenv()

# 1. Carrega credenciais (podem estar no .env ou variáveis de ambiente)
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")

openai_client: Optional[OpenAI] = None
supabase = None
if _OPENAI_KEY:
    try:
        openai_client = OpenAI(api_key=_OPENAI_KEY)
    except Exception:
        openai_client = None

if _SUPABASE_URL and _SUPABASE_KEY:
    try:
        supabase = create_client(_SUPABASE_URL, _SUPABASE_KEY)
    except Exception:
        supabase = None


def extrair_e_enviar_pdf(caminho_do_pdf: str):
    print(f"📖 Lendo o arquivo: {caminho_do_pdf}...")
    try:
        reader = PdfReader(caminho_do_pdf)
    except Exception as e:
        print(f"❌ Não foi possível abrir o PDF: {e}")
        return

    # Percorre cada página do livro PDF
    for num_pagina, pagina in enumerate(reader.pages):
        try:
            texto_pagina = pagina.extract_text() or ""
        except Exception:
            texto_pagina = ""

        # Divide a página em parágrafos — aceita quebras simples ou duplas
        paragrafos = [p.strip() for p in texto_pagina.split("\n\n") if p.strip()] or [p.strip() for p in texto_pagina.split("\n") if p.strip()]

        for paragrafo in paragrafos:
            # Ignora linhas vazias ou muito curtas (como números de página)
            if len(paragrafo) < 40:
                continue

            if not openai_client or not supabase:
                print("⚠️ OpenAI ou Supabase não configurados — pulando envio (modo dry-run).")
                print(f"Conteúdo amostra: {paragrafo[:120]}...\n")
                continue

            try:
                # Transforma o parágrafo científico em um vetor (números)
                resposta_vetor = openai_client.embeddings.create(
                    input=paragrafo,
                    model="text-embedding-3-small"
                )
                vetor = resposta_vetor.data[0].embedding

                # Salva o texto e o vetor na tabela que criamos no Supabase
                supabase.table("documentos_cientificos").insert({
                    "conteudo": paragrafo,
                    "embedding": list(vetor)
                }).execute()

                print(f"✅ Parágrafo da pág. {num_pagina + 1} enviado com sucesso!")

            except Exception as e:
                print(f"❌ Erro ao enviar parágrafo: {e}")


# --- EXECUÇÃO DO SCRIPT AUTOMÁTICA ---
# Este bloco varre a pasta procurando QUALQUER arquivo que termine com .pdf

pasta_atual = os.getcwd()
arquivos_na_pasta = os.listdir(pasta_atual)

# Filtra apenas os arquivos que são PDFs (case-insensitive)
pdfs_encontrados = [arquivo for arquivo in arquivos_na_pasta if arquivo.lower().endswith('.pdf')]

if len(pdfs_encontrados) == 0:
    print("⚠️ Nenhum arquivo PDF encontrado na pasta. Coloque seus livros aqui dentro!")
else:
    print(f"📚 Encontrei {len(pdfs_encontrados)} arquivo(s) PDF para processar.")

    # Roda o processo para cada PDF encontrado
    for pdf in pdfs_encontrados:
        print(f"\n🚀 Iniciando processamento do arquivo: {pdf}")
        extrair_e_enviar_pdf(pdf)
        print(f"✨ Concluído o envio do arquivo: {pdf}\n")

    print("🎉 Processamento concluído (alguns envios podem ter sido pulados em dry-run).")