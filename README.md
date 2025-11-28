# Django REST Framework 템플릿

해커톤 및 프로젝트 시작을 위한 Django REST Framework 템플릿입니다.

## 주요 기능

- ✅ Django 5.2 + Django REST Framework
- ✅ JWT 기반 인증 (쿠키 방식)
- ✅ 사용자 인증 및 관리 (회원가입, 로그인, OAuth)
- ✅ 2FA (TOTP) 지원
- ✅ WebAuthn 지원
- ✅ API 문서 (Swagger/ReDoc)
- ✅ Redis 캐싱
- ✅ Celery 비동기 작업
- ✅ PostgreSQL 데이터베이스
- ✅ Docker 지원

## Vision 모듈 파이프라인

1. **이미지 업로드 & 비전 인식**  
   `POST /api/v1/vision/foods/recognize/`는 Base64 이미지를 정규화하고 SHA256 해시로 30분 캐시에 저장합니다. GPT-4o → GPT-4o-mini 폴백 전략으로 OpenAI Vision을 호출하며 JSON 응답만 허용합니다. 결과는 `FoodRecognitionLog`에 food_name/신뢰도/이미지 플레이스홀더와 함께 기록됩니다.

2. **RAG 안전 가이드**  
   nutrition_pdfs 디렉터리의 PDF를 PyPDFLoader → RecursiveCharacterTextSplitter로 분할해 FAISS(우선) 또는 Chroma 인덱스를 구성하고 LangChain RetrievalQAWithSourcesChain + ChatOpenAI로 {safety_summary,is_safe,nutritional_advice} JSON을 반환합니다. 사용자 임신 주차, 단계, 화법 프롬프트를 포함한 SHA256 캐시 키로 30분간 재사용합니다.

3. **섭취 로그 & 영양 분석**  
   최근 7일 `FoodLog`를 select_related로 불러와 `Food.nutritional_info × portion`을 합산하고 `NutrientRequirement`와 비교합니다. 결과는 1시간 캐시에 저장되며 `GET /vision/food-logs/nutrient-analysis/`로 제공합니다.

4. **개인화 추천**  
   부족 영양소(<70%)를 찾아 `nutritional_info__has_any_keys` 쿼리로 활성 Food를 선별하고 `FoodRecommendation`을 idempotent하게 생성합니다. 최대 5개 추천을 priority 1~10에 매핑하고 1시간 캐시에 묶어 `GET /vision/food-recommendations/personalized/`에서 제공합니다.

5. **후속 상호작용**  
   `FoodRating` CRUD와 `/food-ratings/summary/`는 사용자 평가와 평균, 고/저평점 비율을 제공합니다. `ResponseStyle` CRUD 및 `response-styles/preference/`로 사용자 화법을 업데이트하면 Vision/RAG 파이프라인 전체에 반영됩니다.

## 시작하기

### 필수 요구사항

- Python 3.11+
- PostgreSQL 15+
- Redis
- Docker & Docker Compose (선택사항)

### 로컬 개발 환경 설정

1. **저장소 클론**

```bash
git clone <repository-url>
cd drf-template
```

2. **가상환경 생성 및 활성화**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **의존성 설치**

```bash
pip install -r requirements.txt
```

4. **환경 변수 설정**

`.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 값 설정
```

5. **데이터베이스 마이그레이션**

```bash
python manage.py makemigrations
python manage.py migrate
```

6. **슈퍼유저 생성**

```bash
python manage.py createsuperuser
```

7. **개발 서버 실행**

```bash
# 개발 환경 (간단한 실행)
python manage.py runserver

# 또는 Uvicorn ASGI 서버 사용 (프로덕션과 동일)
uvicorn Reporch.asgi:application --host 0.0.0.0 --port 8000 --reload
```

서버는 `http://localhost:8000`에서 실행됩니다.

### Docker를 사용한 실행

1. **Docker Compose로 전체 스택 실행**

```bash
docker-compose up -d
```

2. **마이그레이션 실행**

```bash
docker-compose exec web python manage.py migrate
```

3. **슈퍼유저 생성**

```bash
docker-compose exec web python manage.py createsuperuser
```

## 환경 변수

`.env.example` 파일을 참고하여 필요한 환경 변수를 설정하세요:

- `SECRET_KEY`: Django 시크릿 키
- `DEBUG`: 디버그 모드 (True/False)
- `POSTGRES_DB`: 데이터베이스 이름
- `POSTGRES_USER`: 데이터베이스 사용자
- `POSTGRES_PASSWORD`: 데이터베이스 비밀번호
- `REDIS_HOST`: Redis 호스트
- `REDIS_PORT`: Redis 포트
- 기타 OAuth 설정 (선택사항)

## API 엔드포인트

### 인증

- `POST /api/v1/auth/` - 회원가입
- `POST /api/v1/auth/` - 로그인
- `POST /api/v1/auth/logout` - 로그아웃
- `GET /api/v1/users/@me/` - 현재 사용자 정보
- `GET /api/v1/profile/<username>/` - 공개 프로필 조회

### API 문서

- `GET /api/schema/swagger/` - Swagger UI
- `GET /api/schema/redoc/` - ReDoc
- `GET /api/schema/` - OpenAPI 스키마

## 실시간 민원 챗봇 (WebSocket)

- **엔드포인트**: `wss://<host>/ws/civil/chat/`
- **인증**: HTTP 쿠키(`accessToken`) 또는 `Authorization: Bearer <token>` 헤더 자동 재사용
- **요청 형식**

```json
{
  "action": "chat",
  "payload": {
    "query": "전입신고 절차 알려줘",
    "session_id": "<optional-uuid>",
    "profile": "student",
    "address": "부산 부산진구 전포동",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
}
```

- **응답 형식**

```json
{
  "type": "chat.response",
  "payload": {
    "session_id": "...",
    "reply": "상세 안내",
    "checklist": [],
    "regulation_sources": [],
    "office": {},
    "recommended_civil_actions": []
  }
}
```

`action: "ping"`을 전송하면 연결 상태를 확인할 수 있으며, 유효성 오류는 `chat.validation_error` 타입으로 수신됩니다.

## 프로젝트 구조

```
drf-template/
├── Reporch/              # 프로젝트 설정
│   ├── settings.py       # Django 설정
│   ├── urls.py           # URL 라우팅
│   └── middlewares/      # 커스텀 미들웨어
├── Users/                # 사용자 앱
│   ├── models.py        # 사용자 모델
│   ├── views.py         # API 뷰
│   ├── serializers.py   # DRF 시리얼라이저
│   └── authentication.py # 인증 백엔드
└── manage.py            # Django 관리 스크립트
```

## 기능 확장

이 템플릿은 기본적인 사용자 인증 기능만 제공합니다. 필요한 기능을 추가하려면:

1. 새로운 Django 앱 생성: `python manage.py startapp <app_name>`
2. `Reporch/settings.py`의 `INSTALLED_APPS`에 추가
3. `Reporch/urls.py`에 URL 패턴 추가

## 테스트

```bash
# 테스트 실행
python manage.py test

# 커버리지와 함께 실행
coverage run --source='.' manage.py test
coverage report
```

## 프로덕션 배포

프로덕션 환경에서는 다음 사항을 확인하세요:

1. `DEBUG=False` 설정
2. `SECRET_KEY`를 안전하게 관리
3. HTTPS 사용
4. 적절한 `ALLOWED_HOSTS` 설정
5. 데이터베이스 백업 설정
6. 정적 파일 서빙 설정 (WhiteNoise 또는 S3)

## 라이선스

MIT License

## 기여

이슈 및 PR을 환영합니다!

