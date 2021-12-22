FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

WORKDIR /app
COPY    . .

# RUN apt-get update && apt-get install -y git
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# EXPOSE 5000
# CMD ["python3", "app.py"]