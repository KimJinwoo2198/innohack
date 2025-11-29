from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import importlib
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

from django.conf import settings
from openai import OpenAI, OpenAIError

LANGCHAIN_COMPONENTS: Dict[str, Any] = {}

from vision.utils.context import build_week_context

logger = logging.getLogger(__name__)

GUIDANCE_LLM_MODEL = getattr(settings, "OPENAI_GUIDANCE_MODEL", "gpt-4o")
GUIDANCE_RAG_TIMEOUT = getattr(settings, "GUIDANCE_RAG_TIMEOUT", 2.5)
GUIDANCE_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "pregnancy_food_guidance",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "safety_summary": {
                "type": "string",
                "description": "사용자 맥락(임산부/일반인)에 맞춰 자연스럽게 설명하는 한국어 안전성 요약(접두사·라벨 금지)",
            },
            "is_safe": {"type": "boolean"},
            "nutritional_advice": {
                "type": "string",
                "description": "사용자 맥락에 맞춘 섭취 가이드와 의학·영양 근거 설명(임산부라면 대한산부인과학회·ACOG·WHO, 일반인이라면 FAO/식약처/WHO 등 공신력 있는 근거 포함)",
            },
        },
        "required": ["safety_summary", "is_safe", "nutritional_advice"],
        "additionalProperties": False,
    },
}

CHAT_HISTORY_LIMIT = getattr(settings, "FOOD_CHAT_HISTORY_LIMIT", 6)
CHAT_REFERENCE_LIMIT = getattr(settings, "FOOD_CHAT_REFERENCE_LIMIT", 3)
CHAT_LLM_MODEL = getattr(settings, "OPENAI_FOOD_CHAT_MODEL", GUIDANCE_LLM_MODEL)
CHAT_OUTPUT_SCHEMA = {
    "type": "json_schema",
    "name": "pregnancy_food_chat_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "한국어 존댓말 서술. 질문자가 임산부인지 일반인인지에 맞춰 위험 성분·감염원·임상 수치·조리법 효과(가열/비가열에 따른 영양소 변화, 미생물 제거, 독소 변화 등) 등 구체 근거를 포함해 답변.",
            },
        },
        "required": ["answer"],
        "additionalProperties": False,
    },
}

FOLLOWUP_PROMPT_TEMPLATE = """너는 대한민국 식품 안전 및 영양 전문가이자 따뜻한 상담사이다.
food_name: {food_name}
임신 맥락(JSON): {week_context}
audience_profile: {audience_profile}
기본 안전 요약: {safety_summary}
기본 영양 조언: {nutritional_advice}

사용자는 임산부일 수도 있고 일반 소비자일 수도 있다. audience_profile에 따라 어조와 근거를 조절한다.
- audience_profile == "pregnancy": 임산부와 태아 안전을 중심으로 설명하고, 대한산부인과학회·ACOG·WHO 등 산모 가이드라인이나 임신 관련 연구를 인용한다.
- audience_profile == "general": 일반 성인·가족 식단 기준으로 설명하며 임산부 표현은 사용하지 않는다. 식약처, WHO, FAO, 국제 영양학회 등 공신력 있는 근거를 인용한다.

기존 대화:
{chat_history}

문헌 스니펫:
{doc_snippets}

사용자 질문:
{question}

지침:
1. 질문의 의도를 파악하고 자유롭게 응답한다.
   - 음식/영양/조리 관련: 전문 지식으로 답변
   - 일상 인사 (안녕, 뭐해 등): 친근하고 따뜻한 톤으로 응답한 뒤 "{food_name}에 대해 궁금한 점이 있으신가요?" 정도로 자연스럽게 이어간다.
   - 기타 일반 질문: 관심을 보이되 음식 상담 영역으로 부드럽게 유도한다.

2. 음식/영양 관련 질문일 때:
   - 존댓말 2~4문장으로 작성하고, audience_profile에 맞는 핵심 안전/영양 포인트를 강조한다.
   - 위험 원인이나 영양학적 메커니즘을 구체적으로 설명하며, 공신력 있는 가이드라인·연구를 문장 안에 자연스럽게 인용한다.
   - 권장/주의 조언과 근거를 하나의 문단으로 서술하고, '근거:' 같은 라벨은 사용하지 않는다.

3. 조리법 관련 질문: 해당 조리법이 영양소 변화, 미생물 제거, 독소 감소/증가, 소화율 등에 미치는 영향을 설명한다.

4. 문헌 스니펫이 있으면 참고하되, 없어도 일반적인 식품학·영양학 지식을 바탕으로 자신감 있게 답변한다.

5. 답변이 명확하지 않거나 전문의 상담이 필요하면 명시한다. 하지만 "근거를 찾지 못했습니다" 같은 부정적 표현은 피한다.
"""

PROMPT_TEMPLATE = """너는 대한민국 식품 안전 및 영양 전문가이다.
dialect_style: {dialect_style}
임신 맥락: {week_context}
audience_profile: {audience_profile}
food_name: {food_name}

제공된 자료(nutrition corpus)를 참고해 해당 음식의 안전성·영양 조언을 JSON 하나로만 출력한다.
- audience_profile == "pregnancy": 임산부와 태아 안전, 산모 대사 변화, 임신 중 주의사항을 중심으로 설명하고 대한산부인과학회·ACOG·WHO 등 산모 가이드라인이나 임신 관련 연구를 인용한다.
- audience_profile == "general": 일반 성인·가족 식단 관점에서 설명하고 임산부 표현을 사용하지 않는다. 식약처, WHO, FAO, 국제 영양학회 등 공신력 있는 근거를 인용한다.
- safety_summary는 라벨 없이 자연스러운 존댓말 문장으로 작성하고, 질문 상황에 맞는 핵심 위험/장점을 한 문단으로 요약한다.
- nutritional_advice는 "권장/주의 문장 + 구체적 이유" 구조를 따르며, 해당 audience_profile에 적합한 공인 근거(가이드라인·메타분석·무작위대조연구 등)를 문장 안에 자연스럽게 인용한다.
- 모든 문장은 key:value 형식이 아니라 연속된 문장으로 서술하고, 과도한 공포 조장은 피한다.
각 필드는 사용자에게 직접 말하듯 정중한 존댓말로 작성하며, 전체 응답은 3줄 이내로 요약한다.
JSON 스키마:
{{
  "safety_summary": "한국어 설명",
  "is_safe": true or false,
  "nutritional_advice": "한국어 설명"
}}

응답 본문은 JSON 객체 외 다른 텍스트를 포함하지 않는다.
"""

openai_client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


def _resolve_audience_profile(week_context: Dict[str, Any]) -> str:
    if isinstance(week_context, dict) and week_context.get("current_week"):
        return "pregnancy"
    return "general"


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
_pipeline_state = {"initialized": False, "initializing": False}
_rag_executor = ThreadPoolExecutor(max_workers=2)


def _preload_pipeline() -> None:
    try:
        vectorstore = _build_vectorstore()
        if vectorstore:
            _build_chain()
            with _pipeline_lock:
                _pipeline_state["initialized"] = True
            logger.info("Guidance pipeline preloaded successfully.")
        else:
            logger.warning("Guidance pipeline preload skipped: vectorstore unavailable.")
    except (RuntimeError, ValueError, OSError, ImportError) as exc:  # pragma: no cover - startup issues
        logger.exception("Guidance pipeline preload failed: %s", exc)
    finally:
        with _pipeline_lock:
            _pipeline_state["initializing"] = False


def initialize_guidance_pipeline():
    with _pipeline_lock:
        if _pipeline_state["initialized"] or _pipeline_state["initializing"]:
            return
        _pipeline_state["initializing"] = True
    Thread(target=_preload_pipeline, name="guidance-preloader", daemon=True).start()


initialize_guidance_pipeline()


def _run_chain_query(chain, question: str) -> Dict[str, Any]:
    # LangChain 0.1.0+ 호환: chain.invoke() 사용
    result = chain.invoke({"question": question})
    raw = result.get("answer") or ""
    return json.loads(raw)


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


def _format_chat_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return "이전 대화 없음"
    trimmed = history[-CHAT_HISTORY_LIMIT:]
    lines: List[str] = []
    for turn in trimmed:
        role = (turn.get("role") or "").lower()
        message = (turn.get("message") or "").strip()
        if not message:
            continue
        speaker = "사용자" if role == "user" else "AI"
        lines.append(f"{speaker}: {message}")
    return "\n".join(lines) if lines else "이전 대화 없음"


def _retrieve_supporting_snippets(food_name: str, question: str, limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    vectorstore = _build_vectorstore()
    if not vectorstore:
        return []
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": limit})
        # 질문 전체를 우선시하고, 음식명과 조리법 키워드를 함께 검색
        # 조리법 관련 키워드가 있으면 더 강조
        cooking_keywords = ["구워", "삶", "튀기", "가열", "조리", "요리", "날것", "생식", "익혀", "데치", "볶", "찜"]
        has_cooking_context = any(keyword in question for keyword in cooking_keywords)
        if has_cooking_context:
            # 조리법 질문: 질문 전체를 우선, 음식명도 포함
            search_query = f"{question} {food_name} 조리법 가열"
        else:
            # 일반 질문: 음식명과 질문을 함께
            search_query = f"{food_name} {question}"
        documents = retriever.get_relevant_documents(search_query)
    except (RuntimeError, ValueError, AttributeError) as exc:  # pragma: no cover - 방어적 로깅
        logger.warning("문헌 스니펫 검색 실패: %s", exc)
        return []
    snippets: List[Dict[str, Any]] = []
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        excerpt = (getattr(doc, "page_content", "") or "").strip()
        if not excerpt:
            continue
        cleaned_excerpt = " ".join(excerpt.split())
        snippets.append(
            {
                "source": str(metadata.get("source") or metadata.get("file_path") or "nutrition_pdfs"),
                "page": metadata.get("page") or metadata.get("page_number"),
                "excerpt": cleaned_excerpt[:600],
            }
        )
    return snippets


def _format_doc_snippets(snippets: List[Dict[str, Any]]) -> str:
    if not snippets:
        return "문헌 스니펫 없음"
    lines: List[str] = []
    for idx, snippet in enumerate(snippets, start=1):
        page = snippet.get("page")
        page_str = f"(p.{page})" if page not in (None, "") else ""
        lines.append(f"[Doc {idx}] {snippet.get('source')} {page_str}: {snippet.get('excerpt')}")
    return "\n".join(lines)


def _build_chat_prompt(
    food_name: str,
    week_context: Dict[str, Any],
    base_guidance: Dict[str, Any],
    history: Optional[List[Dict[str, str]]],
    snippets: List[Dict[str, Any]],
    question: str,
    audience_profile: str,
) -> str:
    return FOLLOWUP_PROMPT_TEMPLATE.format(
        food_name=food_name,
        week_context=json.dumps(week_context, ensure_ascii=False),
        audience_profile=audience_profile,
        safety_summary=base_guidance.get("safety_summary", ""),
        nutritional_advice=base_guidance.get("nutritional_advice", ""),
        chat_history=_format_chat_history(history),
        doc_snippets=_format_doc_snippets(snippets),
        question=question,
    )




def generate_food_chat_reply(
    user,
    food_name: str,
    dialect_style: str,
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    base_guidance: Optional[Dict[str, Any]] = None,
    pregnancy_week: Optional[int] = None,
) -> Dict[str, Any]:
    question_text = (question or "").strip()
    if not question_text:
        raise ValueError("질문이 비어 있습니다.")

    base_guidance = base_guidance or get_food_guidance(user, food_name, dialect_style)
    week_context = build_week_context(user, override_week=pregnancy_week)
    audience_profile = _resolve_audience_profile(week_context)
    snippets = _retrieve_supporting_snippets(food_name, question_text, CHAT_REFERENCE_LIMIT)
    prompt = _build_chat_prompt(
        food_name=food_name,
        week_context=week_context,
        base_guidance=base_guidance,
        history=history,
        snippets=snippets,
        question=question_text,
        audience_profile=audience_profile,
    )

    try:
        response = openai_client.responses.create(
            model=CHAT_LLM_MODEL,
            temperature=0,
            text={"format": CHAT_OUTPUT_SCHEMA},
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        )
        payload = json.loads(_extract_openai_text(response))
        # LLM 응답에 references 없음 (스키마에서 제거)
        # 클라이언트용 메타데이터만 추가
        payload["references"] = []
        payload["retrieved_snippets"] = snippets
        return payload
    except (OpenAIError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Food chat 응답 생성 실패: %s", exc)
        # 실패 시에도 사용자 친화적인 기본 응답 제공
        return {
            "answer": "죄송합니다. 현재 답변을 생성하지 못했습니다. 잠시 후 다시 시도해 주세요.",
            "references": [],
            "retrieved_snippets": snippets,
        }


def get_food_guidance(user, food_name: str, dialect_style: str) -> Dict[str, Any]:
    """기본 음식 가이드 조회 - WebSocket baseline 로딩용 (RAG 제외, 빠른 응답)"""
    try:
        week_context = build_week_context(user)
        audience_profile = _resolve_audience_profile(week_context)
        question = PROMPT_TEMPLATE.format(
            dialect_style=dialect_style,
            week_context=json.dumps(week_context, ensure_ascii=False),
            audience_profile=audience_profile,
            food_name=food_name,
        )
        logger.debug("get_food_guidance: food=%s, audience=%s", food_name, audience_profile)
        # baseline 로딩은 빠르게 LLM만 사용 (RAG 제외)
        result = _generate_guidance_with_llm(food_name, question)
        logger.debug("get_food_guidance 완료: %s", result)
        return result
    except Exception as exc:
        logger.error("get_food_guidance 실패: food=%s, error=%s", food_name, exc, exc_info=True)
        # 최악의 경우 기본 가이드 반환
        return _default_guidance(food_name)


def get_food_safety_info(user, food_name: str, dialect_style: str) -> str:
    guidance = get_food_guidance(user, food_name, dialect_style)
    return guidance.get("safety_summary", "")


def get_nutritional_advice(user, food_name: str, dialect_style: str) -> str:
    guidance = get_food_guidance(user, food_name, dialect_style)
    return guidance.get("nutritional_advice", "")


