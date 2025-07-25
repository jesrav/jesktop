FROM python:3.12-slim-bookworm as base

# Stage 1: Build
FROM base AS builder
WORKDIR /app
ARG UV_VERSION=0.5.8

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential

# Download the latest installer
ADD https://astral.sh/uv/${UV_VERSION}/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy relevant files and folders
COPY pyproject.toml uv.lock README.md ./
COPY jesktop jesktop
COPY data data
COPY app.py app.py

RUN uv sync --frozen

# Stage 2: Final
FROM base AS final
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder app/jesktop /app/jesktop
COPY --from=builder app/data /app/data
COPY --from=builder app/app.py app.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
