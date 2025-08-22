## 최적화 코드 구조(개요)

이 저장소의 프론트엔드 최적화/자동 튜너는 TensorRT 친화적인 ONNX 변환 후보를 생성하고, ORT로 검증한 뒤 trtexec로 엔진을 빌드·프로파일하여 가장 빠른 후보를 선택합니다.

핵심 폴더: `src/tuner/`

- `main.py`
  - 전체 파이프라인 CLI. 후보 생성 → 검증(선택) → trtexec 빌드/프로파일 → 최적 후보 선택 및 결과 저장.
  - 주요 옵션:
    - `--trtexec` trtexec 경로, `--precision` FP16/FP32, `--shapes` (opset>=11에서만 사용),
    - `--simplify` onnx-simplifier 사용, `--skip-validation` ORT 검증 건너뛰기,
    - `--skip-build` TRT 빌드 생략, `--save-plan` 엔진 저장, `--export-profile` 레이어 프로파일 JSON 저장.

- `candidate_generator.py`
  - ONNX GraphSurgeon 기반 후보 생성.
  - 규칙(초기 최소 세트):
    - MatMul→Gemm 치환(가중치가 상수일 때),
    - MatMul K축 정렬 패딩(8/16 정렬; 안전한 경우에만),
    - Simplify 후보(onnx-simplifier) 추가.
  - GraphSurgeon cleanup과 topo-sort로 내보내기 전에 정리. 현재 환경에 맞춰 IR version을 10으로 저장.

- `validator.py`
  - 구조 검사(onnx.checker)와 ONNX Runtime(CPU) 비교로 정합성 확인.
  - 실패 시 이유 문자열 반환(검증 실패로 인한 크래시 방지).

- `trt_builder.py`
  - trtexec 래퍼. `--exportTimes`(1회 실행별 latency 기록)와 선택적 `--exportProfile`(레이어 성능) 지원.
  - opset<11(암묵적 배치) 모델에서는 `--shapes` 플래그를 전달하지 않도록 `pass_shapes` 지원.

- `profiler.py`
  - trtexec `--exportTimes` JSON 파싱과 요약(평균/백분위 등) 유틸.

- `selector.py`
  - 후보별 메트릭에서 평균 지연(meanLatency 또는 mean_ms) → 빌드 시간 순으로 최적 후보 선택.

- `utils.py`
  - 입력/출력 이름·shape 문자열 처리 등 보조 유틸.

## 파이프라인 흐름

1) 후보 생성: 입력 ONNX를 읽어 baseline 사본과 규칙 기반 변형(예: Gemm화, K 정렬, simplification)을 생성합니다.
2) 검증(옵션): ORT로 baseline과 후보를 임의 입력에 대해 비교해 수치 정합성을 확인합니다.
3) 빌드/프로파일: trtexec로 각 후보를 FP16/FP32 등 동일 조건으로 빌드하고 `--exportTimes` JSON에 per-run latency를 저장합니다. `--export-profile` 시 레이어 프로파일 JSON도 저장합니다.
4) 선택: 평균 지연이 가장 낮은 후보를 선택하고, `best.onnx`, `best.json`, `metrics.csv`를 출력 디렉터리에 기록합니다.

출력 디렉터리 구조(예)

- `<outdir>/candidates/` 후보 ONNX들
- `<outdir>/logs/` `*_times.json`(per-run), `*_profile.json`(레이어) 
- `<outdir>/plans/` TensorRT 엔진(plan) 파일(옵션)
- `<outdir>/best.onnx`, `<outdir>/best.json`, `<outdir>/metrics.csv`

## 암묵적 vs 명시적 배치 처리

- ONNX opset<11(암묵적 배치) 모델: trtexec에 `--shapes`를 전달하지 않습니다.
- ONNX opset>=11(명시적 배치) 모델: 입력 이름과 정적 차원으로 `--shapes/min/opt/maxShapes`를 동일하게 전달합니다.

## 실행 방법(요약)

- ResNet50(암묵적 배치 예시):
  - 후보 생성+빌드+프로파일(검증 생략):
    - `python -m tuner.main --onnx <ResNet50.onnx> --outdir <outdir> --trtexec <path-to-trtexec> --precision FP16 --skip-validation --simplify --export-profile`
  - 결과 확인: `<outdir>/logs/*_times.json`에서 평균 지연, `<outdir>/logs/*_profile.json`에서 레이어별 시간 비교.

- 명시적 배치 모델(예: 입력 `X:1x3x224x224`):
  - `--shapes X:1x3x224x224`를 추가합니다(opset>=11일 때만 적용).

## 선택 기준과 보고

- 선택 기준: 평균 지연(meanLatency 또는 mean_ms) 최소, 동률 시 빌드 시간.
- `metrics.csv`: 후보 경로, 리턴코드, 요약 메트릭 기록.
- `best.json`: 선택된 후보와 요약 메트릭.

## 한계와 확장 포인트

- 현재 변환 규칙은 보수적으로 동작합니다. 모델 특성에 따라 추가 규칙(레이아웃 정규화, 패턴 병합, activation folding 등) 확장이 용이합니다.
- 타이밍 캐시 재사용 옵션(`--timingCacheFile` 추가 예정)을 통해 후보 간 tactic 차이를 완화할 수 있습니다.
- ORT 검증은 CPU EP 기준이며, 수치 허용오차는 기본 `1e-4`입니다. 필요 시 옵션화 가능합니다.

## 빠른 체크리스트

- 입력 이름/shape 확인(명시적 배치에서 중요)
- 동일 precision/빌더 옵션으로 후보 간 공정 비교
- 필요 시 `--export-profile`로 레이어 병목 파악
