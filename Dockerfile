FROM runpod/pytorch:3.10-2.0.0-117

SHELL ["/bin/bash", "-c"]

WORKDIR /

# System packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y ffmpeg git && \
    apt-get autoremove -y && apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

# Download all models at build time
COPY builder/download_models.py /download_models.py
RUN python /download_models.py && \
    rm /download_models.py

# Add source code last
ADD src .

CMD ["python", "-u", "/rp_handler.py"]