# syntax=docker/dockerfile:1

# CPU-legacy image: ONNX Runtime 1.17.3 (last release that works on AVX-only CPUs).
FROM rust:1-bookworm AS builder

ARG ORT_VERSION=1.17.3
WORKDIR /build

# Official ORT x64 build; 1.18+ requires AVX2 and would SIGILL on Xeon E5-2650.
# cmake/g++ — для sentencepiece-sys (static, та сама C++ бібліотека, що в core).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl cmake g++ make \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && tar -xzf /tmp/ort.tgz -C /opt \
    && mv "/opt/onnxruntime-linux-x64-${ORT_VERSION}" /opt/ort \
    && rm /tmp/ort.tgz

ENV ORT_STRATEGY=system \
    ORT_LIB_LOCATION=/opt/ort/lib \
    LD_LIBRARY_PATH=/opt/ort/lib

COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --release -p speech-server

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/ort/lib/ /opt/ort/lib/
COPY --from=builder /build/target/release/speech-server /usr/local/bin/speech-server

ENV LD_LIBRARY_PATH=/opt/ort/lib \
    PROMIN_HTTP_ADDR=0.0.0.0:8010 \
    PROMIN_MODELS_DIR=/models \
    PROMIN_THREADS=4 \
    RUST_LOG=info

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/speech-server"]
