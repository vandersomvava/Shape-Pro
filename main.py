from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi import Depends, Header, HTTPException, FastAPI
from pydantic import BaseModel

import os
from typing import List, Optional
from openai import OpenAI, OpenAIError
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = "shapepro_secret_2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

# =========================
# AUTH SYSTEM
# =========================

def gerar_hash(senha: str):
    return pwd_context.hash(senha)


def verificar_senha(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)


def criar_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verificar_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Token ausente")

    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]

    except JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")


# =========================
# MODELS
# =========================

class LoginRequest(BaseModel):
    email: str
    senha: str


class RequisicaoMensagem(BaseModel):
    aluno_id: str
    mensagem: str


class CriarAluno(BaseModel):
    id: str
    nome: Optional[str] = None


# =========================
# CLIENTS
# =========================

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase_client: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)


# =========================
# REGISTER
# =========================
@app.post("/api/v1/register")
async def register_user(user: LoginRequest):
    try:
        senha_hash = gerar_hash(user.senha)

        supabase_client.table("users").insert({
            "email": user.email,
            "senha": senha_hash
        }).execute()

        return {"message": "Usuário criado com sucesso"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# LOGIN
# =========================
@app.post("/api/v1/login")
async def login_user(user: LoginRequest):
    try:
        resp = supabase_client.table("users").select("*").eq("email", user.email).execute()

        if not resp.data:
            raise HTTPException(status_code=401, detail="Usuário não encontrado")

        db_user = resp.data[0]

        if not verificar_senha(user.senha, db_user["senha"]):
            raise HTTPException(status_code=401, detail="Senha incorreta")

        token = criar_token({"sub": db_user["id"]})

        return {"access_token": token, "token_type": "bearer"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# CHAT (CORRIGIDO FALLBACK)
# =========================
@app.post("/api/v1/chat")
async def processar_chat(req: RequisicaoMensagem):
    try:
        resposta_ia = ""

        modelos_tentativa = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

        for modelo in modelos_tentativa:
            try:
                completion = openai_client.chat.completions.create(
                    model=modelo,
                    messages=[
                        {"role": "user", "content": req.mensagem}
                    ],
                    temperature=0.7,
                )

                resposta_ia = completion.choices[0].message.content
                break

            except OpenAIError as e:
                print(f"Erro OpenAI ({modelo}): {e}")
                continue

        if not resposta_ia:
            resposta_ia = "ERRO: Nenhum modelo respondeu."

        return {
            "resposta": resposta_ia,
            "status": "ok"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# ALUNOS
# =========================
@app.post("/api/v1/alunos")
async def criar_aluno(aluno: CriarAluno):
    try:
        supabase_client.table("perfis_alunos").insert({
            "id": aluno.id,
            "nome": aluno.nome,
            "status_fluxo": "ANAMNESE"
        }).execute()

        return {"id": aluno.id, "created": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
