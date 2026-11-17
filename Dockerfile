FROM python:3.12-slim

ENV DATA_ROOT=/input \
    OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app/program/solution
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.sh .
COPY src/ src/
RUN chmod +x run.sh

ENTRYPOINT ["/app/program/solution/run.sh"]
CMD ["--task", "all"]
