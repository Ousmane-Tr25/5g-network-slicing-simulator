FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .

CMD ["python", "-m", "slicing_simulator.cli", "--users", "500", "--output", "results"]
