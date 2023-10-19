FROM python:3.9

# Create and set working directory
WORKDIR /app

# Install Python dependencies
RUN python -m pip install --upgrade pip
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 4200

# Copy the remainder of the code into the image
COPY . ./