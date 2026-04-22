FROM python:3.11-slim

# Create the user according to Hugging Face specs
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy dependency files
COPY --chown=user pyproject.toml poetry.lock* ./

# Install dependencies without virtualenvs setup for container
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy rest of the application
COPY --chown=user src/ ./src/

EXPOSE 7860

# Command to run the application space
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
