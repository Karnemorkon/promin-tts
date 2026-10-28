# syntax=docker/dockerfile:1

# CPU-legacy image: ONNX Runtime 1.31.0 (official x64 build).
# Verified on AVX-only Xeon E5-2650: no AVX2/FMA required for our models.
FROM rust:1-bookworm AS builder

ARG ORT_VERSION=1.31.0
WORKDIR /build

# Official ORT x64 build. 1.31.0 reads IR version 10 (voice_encoder.onnx) and
# still runs on AVX-only CPUs; cmake/g++ are for sentencepiece-sys (static).
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
RUN cargo build --release -p speech-server --bins

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /data/models /data/speakers /data/jobs /data/projects

COPY --from=builder /opt/ort/lib/ /opt/ort/lib/
COPY --from=builder /build/target/release/speech-server /usr/local/bin/speech-server
COPY --from=builder /build/target/release/promin-mcp /usr/local/bin/promin-mcp

ENV LD_LIBRARY_PATH=/opt/ort/lib \
    PROMIN_HTTP_ADDR=0.0.0.0:8010 \
    PROMIN_WYOMING_ADDR=0.0.0.0:10200 \
    PROMIN_MODELS_DIR=/data/models \
    PROMIN_SPEAKERS_DIR=/data/speakers \
    PROMIN_JOBS_DIR=/data/jobs \
    PROMIN_PROJECTS_DIR=/data/projects \
    PROMIN_LOUDNESS_TARGET=-19 \
    PROMIN_TRUE_PEAK=-3 \
    PROMIN_AUTO_DOWNLOAD=auto \
    PROMIN_THREADS=4 \
    RUST_LOG=info

EXPOSE 8010 10200

# curl is intentionally not installed: bash + /dev/tcp keeps the image small.
# The port follows PROMIN_HTTP_ADDR, so a custom listen port keeps working.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["bash", "-c", "addr=\"${PROMIN_HTTP_ADDR:-0.0.0.0:8010}\"; port=\"${addr##*:}\"; exec 3<>/dev/tcp/127.0.0.1/$port && printf 'GET /healthz HTTP/1.0\\r\\n\\r\\n' >&3 && head -n1 <&3 | grep -q '200'"]

ENTRYPOINT ["/usr/local/bin/speech-server"]
