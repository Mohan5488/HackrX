FROM python:3.11

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: Install SQLite CLI
RUN apt-get update && apt-get install -y sqlite3

COPY . .

EXPOSE 9000

CMD ["python", "manage.py", "runserver", "0.0.0.0:9000"]
