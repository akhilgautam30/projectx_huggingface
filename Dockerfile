FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

RUN mkdir /.cache && chmod 777 /.cache

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt
COPY ./weights-roberta-base.h5 /code/weights-roberta-base.h5

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code
COPY . /code/

# Set the command to run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
