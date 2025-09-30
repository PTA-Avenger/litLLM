# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

# Set build arguments
ARG POETRY_VERSION=1.6.1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set poetry configuration
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy poetry files
WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r poetry && useradd -r -g poetry poetry

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs /app/models /app/cache && \
    chown -R poetry:poetry /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    POETRY_LLM_CONFIG_PATH="/app/config/default.yaml" \
    POETRY_LLM_LOG_LEVEL="INFO"

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data')"
ENV NLTK_DATA="/app/nltk_data"

# Switch to non-root user
USER poetry

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from src.stylometric import PoetryLLMSystem; system = PoetryLLMSystem(); exit(0 if system.initialize() else 1)"

# Default command
CMD ["python", "-m", "src.main"]