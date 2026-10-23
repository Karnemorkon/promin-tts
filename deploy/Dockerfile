# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder
WORKDIR /build
COPY Cargo.toml rust-toolchain.toml ./
COPY crates ./crates
RUN cargo build --release -p speech-server

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/speech-server /usr/local/bin/speech-server
ENV PROMIN_HTTP_ADDR=0.0.0.0:8010
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8010/healthz || exit 1
ENTRYPOINT ["/usr/local/bin/speech-server"]
