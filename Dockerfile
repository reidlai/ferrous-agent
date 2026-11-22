# syntax=docker/dockerfile:1

FROM --platform=linux/amd64 rust:1.99-alpine AS builder

# hadolint ignore=DL3018
RUN apk add --no-cache musl-dev ca-certificates binutils \
    && printf 'nonroot:x:10001:10001:nonroot:/:/sbin/nologin\n' > /etc/passwd.scratch \
    && printf 'nonroot:x:10001:\n' > /etc/group.scratch \
    && mkdir -p /tmp_scratch /delta_scratch \
    && chmod 1777 /tmp_scratch \
    && chown 10001:10001 /delta_scratch

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY skills ./skills

RUN RUSTFLAGS="-C target-feature=+crt-static" \
    cargo build --release --locked --target x86_64-unknown-linux-musl --bin ferrous-agent \
    && strip target/x86_64-unknown-linux-musl/release/ferrous-agent

FROM scratch

COPY --from=builder /etc/passwd.scratch /etc/passwd
COPY --from=builder /etc/group.scratch /etc/group
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder --chown=10001:10001 /tmp_scratch /tmp
COPY --from=builder --chown=10001:10001 /delta_scratch /delta
COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/ferrous-agent /ferrous-agent

USER 10001:10001

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["/ferrous-agent", "--healthcheck"]

ENTRYPOINT ["/ferrous-agent"]
CMD ["serve"]
