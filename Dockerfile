FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hosts such as Render set PORT; the app listens on 0.0.0.0 when PORT is set.
ENV PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
