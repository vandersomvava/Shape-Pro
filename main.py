from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

SECRET_KEY = "shapepro_secret_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =========================
# AUTH SYSTEM
# =========================

def gerar_hash(senha: str):
    return pwd_context.hash(senha)


def verificar_senha(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)


def criar_token(data: dict):
    to_encode = data.copy()

    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verificar_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Token ausente")

    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")

    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")


import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import socket
from openai import OpenAI, OpenAIError
from supabase import create_client, Client
from dotenv import load_dotenv
try:
    from postgrest.exceptions import APIError
except Exception:
    APIError = Exception

load_dotenv()

app = FastAPI(title="ShapePro AI Engine - Versão Científica")

# =========================
# MODELOS
# =========================

class CadastroRequest(BaseModel):
    nome: str
    email: str
    senha: str


class LoginRequest(BaseModel):
    email: str
    senha: str


# =========================
# ROTAS
# =========================

@app.post("/api/v1/auth/cadastro")
async def cadastrar_profissional(req: CadastroRequest):

    existente = supabase_client.table("profissionais") \
        .select("id") \
        .eq("email", req.email) \
        .execute()

    if existente.data:
        raise HTTPException(status_code=400, detail="E-mail já cadastrado.")

    senha_cripto = gerar_hash(req.senha)

    novo_prof = supabase_client.table("profissionais").insert({
        "nome": req.nome,
        "email": req.email,
        "senha_hash": senha_cripto
    }).execute()

    return {
        "mensagem": "Cadastro realizado com sucesso!",
        "profissional_id": novo_prof.data[0]["id"]
    }


@app.post("/api/v1/auth/login")
async def login_profissional(req: LoginRequest):

    prof = supabase_client.table("profissionais") \
        .select("*") \
        .eq("email", req.email) \
        .execute()

    if not prof.data:
        raise HTTPException(status_code=401, detail="E-mail ou senha incorretos.")

    profissional = prof.data[0]

    if not verificar_senha(req.senha, profissional["senha_hash"]):
        raise HTTPException(status_code=401, detail="E-mail ou senha incorretos.")

    token = criar_token({
        "sub": str(profissional["id"]),
        "email": profissional["email"]
    })

    return {
        "access_token": token,
        "token_type": "bearer",
        "profissional": {
            "nome": profissional["nome"],
            "plano": profissional.get("plano", "Basico")
        }
    }
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa clientes com checagem de variáveis de ambiente
_openai_key = os.getenv("OPENAI_API_KEY")
openai_client = None
if _openai_key:
    try:
        openai_client = OpenAI(api_key=_openai_key)
    except Exception:
        openai_client = None

_supabase_url = os.getenv("SUPABASE_URL")
_supabase_key = os.getenv("SUPABASE_KEY")
supabase_client: Optional[Client] = None
if _supabase_url and _supabase_key:
    try:
        supabase_client = create_client(_supabase_url, _supabase_key)
    except Exception:
        supabase_client = None

class RequisicaoMensagem(BaseModel):
    aluno_id: str
    mensagem: str


class CriarAluno(BaseModel):
    id: str
    nome: Optional[str] = None


class RespostaChat(BaseModel):
    resposta: str
    status_fluxo: str

PROMPT_ORQUESTRADOR = """Você é o Treinador Central da plataforma SHAPE PRO. Seu objetivo é fazer uma anamnese rápida. 
Pergunte uma coisa por vez: Objetivo, Experiência e Lesões. Não monte treino até ter tudo. 
Se tiver tudo, termine a frase estritamente com a tag [STATUS: PRONTO_PARA_TREINO]."""

# O Prompt do treinador agora virou uma função para aceitar a base científica dinamicamente
def gerar_prompt_treinador(contexto_cientifico_pdf):
    return f"""Você é o Treinador de Elite da SHAPE PRO. Monte a ficha de treino do aluno baseando-se nas lesões e objetivos informados.
    
    ATENÇÃO: Você deve basear suas condutas técnicas estritamente nesta literatura científica oficial extraída dos nossos livros de cabeceira:
    {contexto_cientifico_pdf}
    
    Sempre use tabelas organizadas em Markdown para os exercícios e justifique brevemente com base na ciência enviada acima."""


def ensure_user_exists(user_id: str):
    """Ensure a row exists in the `users` table for foreign key constraints."""
    try:
        if not supabase_client:
            return
        # Check if 'users' table exists in the schema; if not, skip gracefully
        try:
            exists = supabase_client.table("users").select("id").limit(1).execute()
        except APIError as ae:
            # table may not exist in public schema (skip best-effort)
            return
        if not exists or not getattr(exists, "data", None):
            supabase_client.table("users").insert({"id": user_id}).execute()
    except Exception:
        # Best-effort: don't break the request flow if this fails
        return

@app.post("/api/v1/chat")
async def processar_chat(req: RequisicaoMensagem):
    try:
        # Validar clientes configurados
        if not supabase_client:
            raise HTTPException(status_code=400, detail="SUPABASE_URL and SUPABASE_KEY are not configured")
        if not openai_client:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured")
        # 1. Recupera o perfil do aluno (cria se não existir)
        perfil = supabase_client.table("perfis_alunos").select("*").eq("id", req.aluno_id).execute()
        if not perfil or not getattr(perfil, "data", None):
            # Ensure the referenced user exists to satisfy foreign key constraints
            ensure_user_exists(req.aluno_id)
            try:
                supabase_client.table("perfis_alunos").insert({"id": req.aluno_id, "status_fluxo": "ANAMNESE"}).execute()
            except Exception as e:
                msg = str(e)
                if 'foreign key' in msg or 'violates foreign key constraint' in msg:
                    raise HTTPException(status_code=500, detail=("Foreign key constraint prevents creating perfil. "
                                                               "Ensure a corresponding user row exists in your DB or adjust the schema."))
                raise
            status_atual = "ANAMNESE"
        else:
            status_atual = perfil.data[0].get("status_fluxo", "ANAMNESE")

        # 2. Salva a nova mensagem do usuário no banco
        supabase_client.table("historico_chat").insert({"aluno_id": req.aluno_id, "role": "user", "content": req.mensagem}).execute()

        # 3. Busca todo o histórico de conversas antigas deste aluno
        historico_db = supabase_client.table("historico_chat").select("role", "content").eq("aluno_id", req.aluno_id).order("created_at").execute()
        
        # --- 🚀 AQUI ACONTECE O PASSO 3 (A MÁGICA DO RAG) ---
        contexto_cientifico = ""
        
        # Se o aluno já passou da anamnese e está na hora de passar o treino, buscamos nos PDFs
        if status_atual == "PRONTO_PARA_TREINO":
            # A. Transforma a pergunta atual do aluno em um vetor numérico
            try:
                emb_resp = openai_client.embeddings.create(input=req.mensagem, model="text-embedding-3-small")
                vetor_pergunta = emb_resp.data[0].embedding
            except Exception:
                vetor_pergunta = None

            # B. Se conseguimos embedding, consultamos o supabase (RPC deve existir no banco)
            documentos_esportivos = None
            if vetor_pergunta:
                # garante serialização (lista nativa)
                try:
                    argumentos = {
                        "query_embedding": list(vetor_pergunta),
                        "match_threshold": 0.3,
                        "match_count": 2,
                    }
                    documentos_esportivos = supabase_client.rpc("buscar_documentos", argumentos).execute()
                except Exception:
                    documentos_esportivos = None

            # C. Junta os textos científicos encontrados em uma única variável
            if documentos_esportivos and getattr(documentos_esportivos, "data", None):
                for doc in documentos_esportivos.data:
                    contexto_cientifico += doc.get("conteudo", "") + "\n"

        # 4. Define o prompt correto com base no momento do aluno
        if status_atual == "PRONTO_PARA_TREINO":
            prompt_sistema = gerar_prompt_treinador(contexto_cientifico)
        else:
            prompt_sistema = PROMPT_ORQUESTRADOR
        
        # Monta a carga de mensagens para enviar para a OpenAI
        mensagens_ia = [{"role": "system", "content": prompt_sistema}]
        historico_msgs = historico_db.data if getattr(historico_db, "data", None) else []
        for msg in historico_msgs:
            mensagens_ia.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

                # 5. Executa a Inteligência Artificial com fallback de modelos
        resposta_ia = ""

        modelos_tentativa = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

        for modelo in modelos_tentativa:
            try:
                print(f"Tentando modelo: {modelo}")

                completion = openai_client.chat.completions.create(
                    model=modelo,
                    messages=mensagens_ia,
                    temperature=0.2,
                )

                escolha = completion.choices[0]
                resposta_ia = escolha.message.content

                print(f"Resposta recebida do modelo {modelo}")
                break

            except OpenAIError as e:
                print(f"ERRO OPENAI ({modelo}): {e}")

            except Exception as e:
                print(f"ERRO GERAL ({modelo}): {e}")

        if not resposta_ia:
            resposta_ia = "ERRO: Nenhum modelo da OpenAI conseguiu responder."

        # 6. Atualiza o status do fluxo se a IA deu o sinal verde
        if resposta_ia and "[STATUS: PRONTO_PARA_TREINO]" in resposta_ia:
            status_atual = "PRONTO_PARA_TREINO"
            resposta_ia = resposta_ia.replace("[STATUS: PRONTO_PARA_TREINO]", "").strip()
            supabase_client.table("perfis_alunos").update({"status_fluxo": "PRONTO_PARA_TREINO"}).eq("id", req.aluno_id).execute()

        # 7. Salva a resposta da IA no Banco de Dados antes de mandar para a tela do site
        supabase_client.table("historico_chat").insert({"aluno_id": req.aluno_id, "role": "assistant", "content": resposta_ia}).execute()

        return RespostaChat(resposta=resposta_ia, status_fluxo=status_atual)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Certifique-se de importar o Depends no topo do arquivo caso não esteja: de fastapi import Depends
@app.post("/api/v1/alunos")
async def criar_aluno(aluno: CriarAluno, profissional_id: str = Depends(verificar_token)):
    try:
        if not supabase_client:
            raise HTTPException(status_code=400, detail="SUPABASE_URL and SUPABASE_KEY are not configured")
        
        # 1. Verifica se o aluno já existe
        resp = supabase_client.table("perfis_alunos").select("id").eq("id", aluno.id).execute()
        if resp and getattr(resp, "data", None):
            return {"id": aluno.id, "created": False}
        
        # 2. [Opcional] Se você removeu a foreign key rígida do auth.users, pode comentar a linha abaixo:
        # ensure_user_exists(aluno.id)
        
        # 3. Insere o aluno vinculando-o diretamente ao ID do profissional logado!
        try:
            supabase_client.table("perfis_alunos").insert({
                "id": aluno.id, 
                "nome": aluno.nome or "", 
                "status_fluxo": "ANAMNESE",
                "profissional_id": profissional_id  # <--- Vinculação mágica aqui!
            }).execute()
        except Exception as e:
            msg = str(e)
            if 'foreign key' in msg or 'violates foreign key constraint' in msg:
                raise HTTPException(status_code=500, detail=(
                    "Erro de Chave Estrangeira. Garanta que a coluna profissional_id "
                    "está criada corretamente no Supabase."
                ))
            raise
            
        return {"id": aluno.id, "created": True, "profissional_id": profissional_id}
        
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e)
        if 'getaddrinfo' in msg or 'Name or service not known' in msg:
            raise HTTPException(status_code=400, detail="Could not resolve Supabase host. Check SUPABASE_URL.")
        raise HTTPException(status_code=500, detail=msg)
