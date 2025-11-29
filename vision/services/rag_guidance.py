from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
import importlib
from threading import Lock
from typing import Any, Dict

from django.conf import settings
from openai import OpenAI, OpenAIError

LANGCHAIN_COMPONENTS: Dict[str, Any] = {}

from vision.utils.context import build_week_context

logger = logging.getLogger(__name__)

GUIDANCE_LLM_MODEL = getattr(settings, "OPENAI_GUIDANCE_MODEL", "gpt-4o")
GUIDANCE_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "pregnancy_food_guidance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "safety_summary": {
                "type": "string",
                "description": "임산부 음식 안전성에 대한 한국어 요약",
            },
            "is_safe": {"type": "boolean"},
            "nutritional_advice": {
                "type": "string",
                "description": "임산부 영양/섭취 가이드와 반드시 포함되는 의학·영양적 근거 설명",
            },
        },
        "required": ["safety_summary", "is_safe", "nutritional_advice"],
        "additionalProperties": False,
    },
}

PROMPT_TEMPLATE = """너는 대한민국 임산부 안전 및 영양 전문가이다.
dialect_style: {dialect_style}
임신 맥락: {week_context}
food_name: {food_name}

제공된 자료(nutrition corpus)를 참고해 해당 음식의 임산부 음식 안전성 여부와 영양 조언을 JSON 하나로만 출력한다.
nutritional_advice 필드는 권장/주의사항 뒤에 반드시 \"근거: ...\" 형식으로 의학적/영양학적 이유를 포함한다.
각 필드는 임산부에게 직접 말하듯 정중한 존댓말로 작성하며, 전체 응답은 3줄 이내로 요약한다.
JSON 스키마:
{{
  "safety_summary": "한국어 설명",
  "is_safe": true or false,
  "nutritional_advice": "한국어 설명"
}}

응답 본문은 JSON 객체 외 다른 텍스트를 포함하지 않는다.
"""

openai_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


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
    llm = chat_cls(model="gpt-4.1-nano", temperature=0)
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


_pipeline_lock = Lock()
_pipeline_state = {"initialized": False}


def initialize_guidance_pipeline():
    if _pipeline_state["initialized"]:
        return
    with _pipeline_lock:
        if _pipeline_state["initialized"]:
            return
        try:
            vectorstore = _build_vectorstore()
            if vectorstore:
                _build_chain()
                _pipeline_state["initialized"] = True
                logger.info("Guidance pipeline preloaded successfully.")
            else:
                logger.warning("Guidance pipeline preload skipped: vectorstore unavailable.")
        except (RuntimeError, ValueError, OSError, ImportError) as exc:  # pragma: no cover - startup issues
            logger.exception("Guidance pipeline preload failed: %s", exc)


initialize_guidance_pipeline()


def _default_guidance(food_name: str) -> Dict[str, Any]:
    return {
        "safety_summary": f"{food_name}에 대한 안전 정보를 확보하지 못했습니다. 담당 전문가와 상의해 주세요.",
        "is_safe": False,
        "nutritional_advice": "확인되지 않은 음식은 섭취량을 제한하고 전문 의료진과 상담하세요.",
    }


def _extract_openai_text(response: Any) -> str:
    output = getattr(response, "output", None)
    if output:
        first = output[0]
        if getattr(first, "content", None):
            first_content = first.content[0]
            if hasattr(first_content, "text"):
                return first_content.text
    raise ValueError("OpenAI Guidance 응답에서 텍스트를 추출하지 못했습니다.")


def _call_guidance_llm(question: str) -> Dict[str, Any]:
    if not getattr(openai_client, "api_key", None):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    response = openai_client.responses.create(
        model=GUIDANCE_LLM_MODEL,
        temperature=0,
        text={"format": GUIDANCE_OUTPUT_SCHEMA},
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": question
                        + "\n참고 문헌이 없다면, 최신 임산부 영양 가이드를 기반으로 답하세요.",
                    }
                ],
            }
        ],
    )
    text = _extract_openai_text(response)
    return json.loads(text)


def _generate_guidance_with_llm(food_name: str, question: str) -> Dict[str, Any]:
    try:
        return _call_guidance_llm(question)
    except (OpenAIError, ValueError, json.JSONDecodeError) as exc:
        logger.error("LLM 폴백 가이드 생성 실패: %s", exc)
        return _default_guidance(food_name)


def get_food_guidance(user, food_name: str, dialect_style: str) -> Dict[str, Any]:
    week_context = build_week_context(user)
    chain = _build_chain()
    question = PROMPT_TEMPLATE.format(
        dialect_style=dialect_style,
        week_context=json.dumps(week_context, ensure_ascii=False),
        food_name=food_name,
    )
    if not chain:
        return _generate_guidance_with_llm(food_name, question)

    try:
        result = chain({"question": question})
        raw = result.get("answer") or ""
        guidance = json.loads(raw)
        return guidance
    except json.JSONDecodeError as exc:
        logger.error("RAG JSON 파싱 실패: %s", exc)
        return _generate_guidance_with_llm(food_name, question)
    except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - RAG runtime issues
        logger.exception("RAG 가이드 생성 실패: %s", exc)
        return _generate_guidance_with_llm(food_name, question)


def get_food_safety_info(user, food_name: str, dialect_style: str) -> str:
    guidance = get_food_guidance(user, food_name, dialect_style)
    return guidance.get("safety_summary", "")


def get_nutritional_advice(user, food_name: str, dialect_style: str) -> str:
    guidance = get_food_guidance(user, food_name, dialect_style)
    return guidance.get("nutritional_advice", "")


