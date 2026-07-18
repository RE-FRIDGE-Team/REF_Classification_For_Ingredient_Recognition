"""
술 브랜드 마이닝 CLI — 학습 데이터에서 집중도 기준으로 술 브랜드 가제티어 생성.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[목적]
  브랜드 집중도 리포트에서 확인된 문제 — 오리지널(0.10)·청정원(0.02)·
  웅진빅토리아(0.33)·김소형원방(0.20) 같은 저집중 브랜드의 술 사전 오염 —
  를 데이터 기반 필터로 원천 차단하는 도구.

  필터 규칙 (기본값):
    - min_support=2       : 술 카테고리 등장 2회 미만 브랜드 배제 (1회 우연 차단)
    - min_concentration=0.8 : 술빈도/전체빈도 0.8 미만 배제
    - ALCOHOL_BRAND_BLOCKLIST : 집중도 무관 상시 배제 (일반어·겸업 브랜드)
  → 하이트진로(0.40, 음료 겸업)도 기본값에서 자동 배제되는 구조.
    (술 전용으로 쓰고 싶으면 --min-concentration 을 낮춰 재실행)

[사용법]
  python tools/mine_alcohol_brands.py --input data.csv \
      --config configs/experiment.yaml \
      --out resources/alcohol_brands.txt

  산출 파일은 src/lexicons.load_alcohol_brands() 가 자동 병합 로드하므로,
  이후 학습 시 별도 조치 불필요.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_utils import load_data
from src.lexicons import mine_alcohol_brands
from src.preprocess import PreprocessOptions, REFPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mine_alcohol_brands")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="술 브랜드 집중도 마이닝 → 가제티어 파일 생성")
    p.add_argument("--input", required=True, help="CSV/XLSX 학습 데이터 경로")
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--out", default="resources/alcohol_brands.txt")
    p.add_argument("--min-support", type=int, default=2,
                   help="술 카테고리 최소 등장 횟수 (미만 배제)")
    p.add_argument("--min-concentration", type=float, default=0.8,
                   help="최소 술 집중도 = 술빈도/전체빈도 (미만 배제)")
    p.add_argument("--alcohol-label", default="술", help="술 대분류 라벨 문자열")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = load_data(
        input_path=args.input,
        col_map=cfg["data"]["input_columns"],
        exclude_large=cfg["data"].get("exclude_large"),
        exclude_tag=cfg["data"].get("exclude_tag"),
    )

    # 브랜드 메타데이터 추출 목적의 경량 전처리 — 형태소 분석 생략으로 고속화
    prep = REFPreprocessor(
        brand_dict_path=cfg["data"].get(
            "brand_dict_path",
            "product_data_collection/not_grocery_and_brand_list/grocery_brand_name.json",
        ),
        options=PreprocessOptions(morpheme_analyzer="none"),
        use_parser=True,
    )
    df = prep.fit_transform(df)

    mined = mine_alcohol_brands(
        df["parsed_brand"], df["large_category"],
        alcohol_label=args.alcohol_label,
        min_support=args.min_support,
        min_concentration=args.min_concentration,
    )

    # 리포트 출력 — 검수 편의를 위한 집중도 테이블
    logger.info("필터 통과 브랜드 %d개 (support≥%d, concentration≥%.2f)",
                len(mined), args.min_support, args.min_concentration)
    print(f"\n{'브랜드':<20s} {'술빈도':>6s} {'전체':>6s} {'집중도':>6s}")
    print("-" * 46)
    for b, n_alc, n_all, conc in mined:
        print(f"{b:<20s} {n_alc:>6d} {n_all:>6d} {conc:>6.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# tools/mine_alcohol_brands.py 자동 생성 — 직접 수정 대신 재실행 권장\n")
        f.write(f"# 필터: support>={args.min_support}, concentration>={args.min_concentration}\n")
        for b, *_ in mined:
            f.write(b + "\n")
    logger.info("가제티어 저장 완료: %s", out)


if __name__ == "__main__":
    main()
