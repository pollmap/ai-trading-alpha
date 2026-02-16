# ATLAS × CUFA NEXUS / CUFA Web 통합 가이드

이 문서는 다음 질문에 답합니다.

- "ATLAS를 CUFA NEXUS/CUFA Web와 **어떻게** 합치는 게 좋은가?"
- "지금 당장 실행 가능한 통합 순서는 무엇인가?"
- "배포 전/후 어떤 리스크를 특히 확인해야 하나?"

---

## 1) 권장 역할 분리 (결론)

가장 안전하고 확장성 높은 구조는 아래 3분할입니다.

1. **ATLAS**: 전략 시뮬레이션/평가 엔진 API
   - 백테스트/시뮬레이션 실행
   - 모델/전략 성능 산출
   - 리포트 데이터 생성
2. **CUFA NEXUS**: 오케스트레이션 + 사용자/권한 + 워크플로우 허브
   - 사용자 조직/프로젝트/권한 관리
   - Job 생성·중단·재시작
   - 외부 서비스 연계(알림, 결제, 정책)
3. **CUFA Web**: 사용자 UI/리포팅
   - 대시보드, 비교 화면, 실행 상태 표시
   - 결과 탐색/다운로드

핵심은 **엔진과 제품 계층을 분리**해 책임을 명확하게 두는 것입니다.

---

## 2) 왜 이 구조가 맞는가

ATLAS는 이미 다음 성격을 갖고 있어 "엔진화"에 유리합니다.

- JWT/API Key 기반 인증 확장 포인트
- 사용량 계측/쿼터 개념
- 시뮬레이션/분석/리포트 기능 분리
- DB 기반 결과 저장 및 조회 흐름

즉, CUFA가 제품/플랫폼 계층을 담당하고, ATLAS가 계산 엔진으로 독립 운영되면
조직/결제/운영 정책 변경에도 코어 시뮬레이션 품질을 안정적으로 유지할 수 있습니다.

---

## 3) 통합 시 가장 중요한 3개 이슈

### A. 인증 주체 충돌 (JWT 이중화)

문제:
- NEXUS와 ATLAS가 각각 JWT를 발급하면, 신뢰 체인이 꼬여 장애 원인이 됩니다.

권장:
- **NEXUS를 단일 IdP(토큰 발급 주체)로 고정**
- ATLAS는 NEXUS 서명 검증만 수행(또는 내부 서비스 토큰만 수용)
- 사용자 토큰과 서비스 간 토큰(머신 토큰)을 분리

실무 팁:
- `iss`, `aud`, `scope`, `tenant_id`를 필수 클레임으로 표준화
- 키 롤링(JWKS) 절차를 문서화

### B. 장기 실행 Job 상태 동기화

문제:
- 시뮬레이션은 장시간 실행될 수 있어, UI에서 "멈춤/완료/실패"가 어긋나기 쉽습니다.

권장:
- "요청-수락-비동기 실행" 3단계로 설계
- NEXUS가 `job_id`를 발급하고 상태를 폴링/이벤트로 수집
- ATLAS는 상태 전이를 엄격히 관리 (`queued -> running -> completed|failed|cancelled`)

실무 팁:
- 상태 변경을 idempotent하게 처리
- 재시작 시 마지막 체크포인트부터 resume

### C. 비용/쿼터 정책 불일치

문제:
- ATLAS의 내부 사용량 계측과 CUFA 과금 정책이 다르면, 고객 불만/정산 오류가 발생합니다.

권장:
- 과금 단위를 미리 단일화:
  - API call
  - simulation minute
  - model-token usage
- NEXUS를 소스 오브 트루스로 두고 ATLAS 계측값을 동기화

실무 팁:
- "예상 비용"과 "실제 비용" 모두 기록
- tenant별 하드리밋/소프트리밋 분리

---

## 4) 추천 통합 아키텍처 (실행형)

```text
[CUFA Web]
    |
    v
[CUFA NEXUS API] --(authN/authZ, tenancy, billing policy, job orchestration)--+
    |                                                                       |
    | REST/Webhook/Event                                                    |
    v                                                                       v
[ATLAS Engine API] --(simulate/backtest/report)--> [Postgres/Redis/Object Storage]
```

요점:
- 외부 클라이언트는 직접 ATLAS를 호출하지 않고 NEXUS를 경유
- ATLAS는 계산/분석에 집중, NEXUS는 정책/권한/오케스트레이션 담당

---

## 5) 단계별 합치기 (현실적인 로드맵)

### Phase 0. 계약 정리 (1~2일)
- API 계약서(OpenAPI) 우선 고정
- 상태코드/에러코드/클레임 스키마 합의
- tenant_id, project_id, job_id 네이밍 규칙 고정

### Phase 1. 최소 통합 (3~5일)
- NEXUS에서 "시뮬레이션 생성" API 추가
- ATLAS `/simulate` 호출 프록시 구현
- Web에서 Job 생성/목록/상태만 노출

### Phase 2. 운영 안정화 (1~2주)
- 취소/재시작/재시도 정책 추가
- 메트릭, 추적ID, 감사로그 연결
- 실패 유형 분류(입력 오류/외부 API 오류/엔진 오류)

### Phase 3. 제품화 고도화 (지속)
- 과금/플랜별 쿼터 실시간 반영
- 시뮬레이션 템플릿/전략 마켓플레이스
- 결과 리포트 자동 생성 및 공유 링크

---

## 6) API 경계 제안

### NEXUS -> ATLAS (내부 API)

- `POST /internal/simulations`
  - 입력: `tenant_id`, 전략 설정, 마켓, cycle, 모델
  - 출력: `job_id`, `status=queued`
- `GET /internal/simulations/{job_id}`
  - 출력: 상태 + 진행률 + 오류코드
- `POST /internal/simulations/{job_id}/cancel`

### Web -> NEXUS (공개 API)

- `POST /v1/workspaces/{workspace_id}/simulations`
- `GET /v1/workspaces/{workspace_id}/simulations`
- `GET /v1/workspaces/{workspace_id}/simulations/{id}`

주의:
- Web는 절대 ATLAS 내부 엔드포인트를 직접 때리지 않게 유지

---

## 7) 데이터 모델 매핑 초안

- **Tenant**: NEXUS 기준 원장
- **Workspace/Project**: NEXUS 도메인
- **SimulationJob**: NEXUS 주도, ATLAS는 execution detail 보유
- **SimulationResult**: ATLAS 생성, NEXUS는 인덱스/요약 저장

권장:
- 결과 원문(JSONL/대형 산출물)은 Object Storage로 분리
- NEXUS DB에는 검색용 메타데이터만 저장

---

## 8) 배포 방법 (친절 버전)

## 8-1. 로컬 개발 배포

1. 저장소 준비
   - `git clone ... && cd ai-trading-alpha`
2. 의존성 설치
   - `pip install -e ".[dev]"`
3. 환경 변수
   - `.env`에 LLM/API/DB 값 주입
4. 인프라 기동
   - `make db-up`
   - `make db-init`
5. ATLAS 실행
   - `python scripts/run_benchmark.py --market US --cycles 5`
6. 대시보드 확인
   - `streamlit run src/dashboard/app.py`

## 8-2. 운영 배포 권장

- NEXUS와 ATLAS를 별도 서비스로 배포
- 내부 통신은 mTLS 또는 private network로 제한
- 로그/메트릭/트레이싱을 공통 스택으로 수집
- 릴리즈는 canary(소수 tenant) 후 전체 확장

---

## 9) "이대로 배포 가능한가?" 체크리스트

- [ ] 인증: 토큰 발급 주체 단일화(issuer 고정)
- [ ] 권한: tenant/workspace 경계 테스트 완료
- [ ] 안정성: 취소/재시도/재시작 시나리오 검증
- [ ] 정합성: 비용 계측 수치가 NEXUS와 일치
- [ ] 가시성: job trace id로 end-to-end 추적 가능
- [ ] 보안: 내부 엔드포인트 비공개, 키 롤링 정책 문서화

---

## 10) 바로 시작할 실천안 (이번 주)

1. NEXUS에서 `CreateSimulation` API 계약 확정
2. ATLAS에 내부용 `/internal/simulations` 엔드포인트 어댑터 추가
3. Web에 "시뮬레이션 실행/상태" 최소 화면 연결
4. 취소/재시작 정책까지 MVP 포함
5. 첫 번째 tenant 1개로 파일럿 운영

위 5개만 먼저 끝내면 "합치는 것"이 개념이 아니라 실제 운영 상태로 들어갑니다.
