# ═══════════════════════════════════════════════════════════════════
# RE:FRIDGE ML — Dockerfile
#
# 목적: JPype1 / KoNLPy를 위한 JDK 내장 Python 환경
# 베이스: python:3.11-slim (multi-arch: linux/amd64 + linux/arm64)
#         → Mac M2 Pro(arm64) / Windows(amd64) 모두 자동 대응
#
# PyCharm 연결:
#   Settings → Python Interpreter → Add → Docker Compose
#   Service: refridge-dev   Python: /usr/local/bin/python
# ═══════════════════════════════════════════════════════════════════

FROM python:3.11-slim

# ── 시스템 패키지 (single RUN layer) ────────────────────────────────
# default-jdk : JPype1 / KoNLPy 런타임 (arm64·amd64 모두 OpenJDK 17)
# fonts-nanum : 한국어 matplotlib 폰트
# git         : pip git+ 패키지 설치용
# build-essential, curl : 일부 C 확장 컴파일 및 헬스체크
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jdk \
    fonts-nanum \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Java 환경 변수 ───────────────────────────────────────────────────
# /usr/lib/jvm/default-java 는 Debian 계열에서
# arm64·amd64 모두 동일하게 존재하는 symlink
ENV JAVA_HOME=/usr/lib/jvm/default-java
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── 작업 디렉토리 ────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python 패키지 설치 (레이어 캐시 활용: requirements만 먼저 복사) ──
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# ── Jupyter 커널 등록 (notebooks/ 사용 시) ──────────────────────────
RUN python -m ipykernel install --name refridge --display-name "RE:FRIDGE (KoNLPy)"

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*


# ── 기본 CMD: PyCharm Docker 인터프리터 연결 시 컨테이너 유지 ────────
# docker-compose의 각 서비스가 command로 override함
CMD ["python"]
