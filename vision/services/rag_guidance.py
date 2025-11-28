from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
import importlib
from typing import Any, Dict

from django.conf import settings

LANGCHAIN_COMPONENTS: Dict[str, Any] = {}

from vision.utils.cache import build_cache_key, get_cache_value, set_cache_value
from vision.utils.context import build_week_context

logger = logging.getLogger(__name__)

GUIDANCE_CACHE_TTL = 1800
PROMPT_TEMPLATE = """너는 대한민국 임산부 안전 및 영양 전문가이다.
dialect_style: {dialect_style}
임신 맥락: {week_context}
food_name: {food_name}

제공된 자료(nutrition corpus)를 참고해 해당 음식의 안전성 여부와 영양 조언을 JSON 하나로만 출력한다.
JSON 스키마:
{{
  "safety_summary": "한국어 설명",
  "is_safe": true or false,
  "nutritional_advice": "한국어 설명"
}}

응답 본문은 JSON 객체 외 다른 텍스트를 포함하지 않는다.
"""


def _ensure_langchain_loaded() -> bool:
    if LANGCHAIN_COMPONENTS:
        return True
    try:
        chains = importlib.import_module("langchain.chains")
        prompts = importlib.import_module("langchain.prompts")
        splitters = importlib.import_module("langchain.text_splitter")
        loaders = importlib.import_module("langchain_community.document_loaders")
        vectors = importlib.import_module("langchain_community.vectorstores")
        openai_mod = importlib.import_module("langchain_openai")
    except ImportError as exc:  # pragma: no cover
        logger.error("LangChain 모듈 로드 실패: %s", exc)
        return False

    LANGCHAIN_COMPONENTS.update(
        {
            "retrieval_chain": chains.RetrievalQAWithSourcesChain,
            "prompt": prompts.PromptTemplate,
            "splitter": splitters.RecursiveCharacterTextSplitter,
            "loader": loaders.PyPDFLoader,
            "faiss": vectors.FAISS,
            "chroma": vectors.Chroma,
            "chat": openai_mod.ChatOpenAI,
            "embedding": openai_mod.OpenAIEmbeddings,
        }
    )
    return True


def _nutrition_docs_path() -> str:
    return os.path.join(settings.BASE_DIR, "nutrition_pdfs")


@lru_cache(maxsize=1)
def _build_vectorstore():
    if not _ensure_langchain_loaded():
        logger.error("LangChain 의존성을 초기화하지 못했습니다.")
        return None
    loader_cls = LANGCHAIN_COMPONENTS.get("loader")
    splitter_cls = LANGCHAIN_COMPONENTS.get("splitter")
    embedding_cls = LANGCHAIN_COMPONENTS.get("embedding")
    faiss_cls = LANGCHAIN_COMPONENTS.get("faiss")
    chroma_cls = LANGCHAIN_COMPONENTS.get("chroma")
    if not all([loader_cls, splitter_cls, embedding_cls]):
        logger.error("필수 LangChain 구성요소가 누락되었습니다.")
        return None
    doc_path = _nutrition_docs_path()
    if not os.path.isdir(doc_path):
        logger.warning("nutrition_pdfs 디렉터리를 찾을 수 없습니다: %s", doc_path)
        return None

    documents = []
    for file_name in os.listdir(doc_path):
        if not file_name.lower().endswith(".pdf"):
            continue
        loader = loader_cls(os.path.join(doc_path, file_name))
        documents.extend(loader.load())

    if not documents:
        logger.warning("nutrition_pdfs 폴더에 처리할 문서가 없습니다.")
        return None

    splitter = splitter_cls(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    embeddings = embedding_cls(model=getattr(settings, "OPENAI_EMBED_MODEL", "text-embedding-3-large"))

    try:
        if not faiss_cls:
            raise RuntimeError("FAISS 모듈이 비어 있습니다.")
        return faiss_cls.from_documents(splits, embeddings)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("FAISS 인덱스 생성 실패, Chroma로 폴백: %s", exc)
        if not chroma_cls:
            logger.error("Chroma 모듈을 찾을 수 없어 인덱스를 생성하지 못했습니다.")
            return None
        return chroma_cls.from_documents(splits, embeddings)


@lru_cache(maxsize=1)
def _build_chain():
    if not _ensure_langchain_loaded():
        logger.error("LangChain 핵심 모듈을 불러올 수 없습니다.")
        return None
    retrieval_cls = LANGCHAIN_COMPONENTS.get("retrieval_chain")
    chat_cls = LANGCHAIN_COMPONENTS.get("chat")
    prompt_cls = LANGCHAIN_COMPONENTS.get("prompt")
    if not all([retrieval_cls, chat_cls, prompt_cls]):
        logger.error("LangChain 구성요소가 비어 있어 체인을 만들 수 없습니다.")
        return None
    vectorstore = _build_vectorstore()
    if not vectorstore:
        return None
    llm = chat_cls(model="gpt-4o", temperature=0)
    prompt = prompt_cls(
        input_variables=["summaries", "question"],
        template="다음 참고자료:\n{summaries}\n\n질문:\n{question}\n\n응답은 반드시 JSON만 포함해야 합니다.",
    )
    return retrieval_cls.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
    )


def _default_guidance(food_name: str) -> Dict[str, Any]:
    return {
        "safety_summary": f"{food_name}에 대한 안전 정보를 확보하지 못했습니다. 담당 전문가와 상의해 주세요.",
        "is_safe": False,
        "nutritional_advice": "확인되지 않은 음식은 섭취량을 제한하고 전문 의료진과 상담하세요.",
    }


def get_food_guidance(user, food_name: str, dialect_style: str) -> Dict[str, Any]:
    week_context = build_week_context(user)
    cache_suffix = f"{food_name}:{dialect_style}:{json.dumps(week_context, sort_keys=True)}"
    cache_key = build_cache_key(user.id, "guidance", cache_suffix)
    cached = get_cache_value(cache_key)
    if cached:
        return cached

    chain = _build_chain()
    if not chain:
        guidance = _default_guidance(food_name)
        set_cache_value(cache_key, guidance, GUIDANCE_CACHE_TTL)
        return guidance

    question = PROMPT_TEMPLATE.format(
        dialect_style=dialect_style,
        week_context=json.dumps(week_context, ensure_ascii=False),
        food_name=food_name,
    )
    try:
        result = chain({"question": question})
        raw = result.get("answer") or ""
        guidance = json.loads(raw)
        set_cache_value(cache_key, guidance, GUIDANCE_CACHE_TTL)
        return guidance
    except json.JSONDecodeError as exc:
        logger.error("RAG JSON 파싱 실패: %s", exc)
        guidance = _default_guidance(food_name)
        set_cache_value(cache_key, guidance, GUIDANCE_CACHE_TTL)
        return guidance
    except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - RAG runtime issues
        logger.exception("RAG 가이드 생성 실패: %s", exc)
        guidance = _default_guidance(food_name)
        set_cache_value(cache_key, guidance, GUIDANCE_CACHE_TTL)
        return guidance


def get_food_safety_info(user, food_name: str, dialect_style: str) -> str:
    guidance = get_food_guidance(user, food_name, dialect_style)
    return guidance.get("safety_summary", "")


def get_nutritional_advice(user, food_name: str, dialect_style: str) -> str:
    guidance = get_food_guidance(user, food_name, dialect_style)
    return guidance.get("nutritional_advice", "")


