FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    libfreetype6-dev \
    libharfbuzz-dev \
    libfontconfig1-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY process_pdfs.py .

RUN mkdir -p /app/input /app/output

RUN chmod +x process_pdfs.py

# Run the application
CMD ["python", "process_pdfs.py"]
