FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc make python3 python3-venv python3-pip libffi-dev curl ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/pypy2.7 \
    && curl -fsSL https://downloads.python.org/pypy/pypy2.7-v7.3.17-linux64.tar.bz2 \
       | tar -xj -C /opt/pypy2.7 --strip-components=1
ENV PATH="/opt/pypy2.7/bin:${PATH}"

WORKDIR /repo
COPY . .

ENV SKIP_WEIGHTS=1
RUN benchmark/paper/setup.sh

ENTRYPOINT ["benchmark/paper/run_all.sh"]
