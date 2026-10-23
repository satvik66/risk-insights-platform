FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY risk_platform ./risk_platform
COPY data ./data
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn","risk_platform.api:app","--host","0.0.0.0","--port","8000"]
