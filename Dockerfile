FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY support_agent/requirements.txt /app/support_agent/requirements.txt
RUN pip install --no-cache-dir -r /app/support_agent/requirements.txt

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "support_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
