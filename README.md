# Sentiment Analyzer

A simple sentiment analysis tool that classifies text as Positive, Negative, or Neutral. It can be run via Streamlit for a web interface or inside a Docker container with CI/CD automation powered by GitHub Actions.

![image alt](https://github.com/mohith-anand/sentiment-analyser/blob/9b1df49c181adda1001c06bc1233845894cad7d5/Screenshot%202025-10-15%20032630.png)
## Features

- Real-time sentiment analysis via Streamlit UI
- Dockerized for easy deployment
- CI/CD automation with GitHub Actions
- Support for multi-sentence input

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- Docker (optional, for containerized deployment)
- Git

## Setup & Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/mohith-anand/sentiment-analyser.git
cd sentiment-analyzer
```

### Step 2: Set Up Python Environment

Create and activate a virtual environment to isolate project dependencies:

**Windows:**
```bash
python -m venv senti_venv
senti_venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv senti_venv
source senti_venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages from the requirements file:

```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Local Streamlit Development

To run the application with Streamlit:

```bash
streamlit run app.py
```

The application will start and be accessible at `http://localhost:8501` in your default web browser.

### Option 2: Docker Deployment

#### Build the Docker Image

Build the Docker image (run this once or after making code changes):

```bash
docker build -t sentiment-analyzer .
```

#### Run the Docker Container

Start the container with port mapping:

```bash
docker run -p 8501:8501 sentiment-analyzer
```

Access the application in your browser at `http://localhost:8501`

## CI/CD Pipeline

The project includes GitHub Actions workflow automation for continuous integration and deployment. The pipeline automatically runs tests and deploys updates when changes are pushed to the repository.

## Troubleshooting

If you encounter any issues:
- Ensure all dependencies in `requirements.txt` are correctly installed
- Verify that port 8501 is not already in use
- For Docker issues, check that Docker is running and properly installed
- Review GitHub Actions logs for deployment errors

