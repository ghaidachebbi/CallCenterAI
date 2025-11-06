You are right, that was a bit lengthy. Here is a **more direct and concise** GitHub README for the CallCenterAI project, focusing only on the essentials and instructions.

## 🧠 CallCenterAI - Intelligent Ticket Classification (MLOps Mini-Project)

-----

### 💡 Project Summary

[cite\_start]This project establishes a **complete MLOps architecture** for automatically classifying customer support tickets[cite: 5, 7]. [cite\_start]It compares two NLP models—**TF-IDF + SVM** and a **Hugging Face Transformer** [cite: 8, 9][cite\_start]—orchestrated by an **Intelligent AI Agent**[cite: 10, 71].

[cite\_start]The system is fully **dockerized**, includes **CI/CD (GitHub Actions)** [cite: 10, 18][cite\_start], and is **monitored** using Prometheus and Grafana[cite: 10, 19].

### 🛠️ Key Technologies

| Category | Tools | Purpose |
| :--- | :--- | :--- |
| **Core** | [cite\_start]Python 3.11, FastAPI [cite: 25] | [cite\_start]Services API Framework[cite: 25]. |
| [cite\_start]**Models** | scikit-learn, Hugging Face [cite: 27, 28] | [cite\_start]TF-IDF/SVM and Transformer models[cite: 27, 28]. |
| **MLOps** | [cite\_start]MLflow, DVC [cite: 30, 31] | [cite\_start]Experiment Tracking/Registry and Data/Pipeline Management[cite: 30, 31]. |
| **Deployment** | [cite\_start]Docker & Docker Compose [cite: 33] | [cite\_start]Conteneurisation and orchestration[cite: 33]. |
| **Automation** | [cite\_start]GitHub Actions [cite: 35] | [cite\_start]CI/CD (Tests, Lint, Build, Push)[cite: 35]. |
| **Observability** | [cite\_start]Prometheus + Grafana [cite: 36] | [cite\_start]Service monitoring[cite: 36]. |

-----

### 🚀 Setup and Launch

#### 1\. Prerequisites

You need **Docker** and **Docker Compose** installed.

#### 2\. Training (Optional)

[cite\_start]The pipeline is defined in `dvc.yaml`[cite: 84]. If you need to re-run the preparation and TF-IDF training:

1.  [cite\_start]Initialize DVC and ensure MLflow is running[cite: 83, 86].
2.  Run the pipeline:
    ```bash
    dvc repro
    ```

#### 3\. Start the Full System

[cite\_start]This command builds and launches all containers (Agent, Services, MLflow, Prometheus, Grafana)[cite: 79, 80, 81]:

```bash
docker-compose up --build -d
```

#### 4\. Access Endpoints

| Service | Port | Description |
| :--- | :--- | :--- |
| **Agent IA Service** | 8000 | [cite\_start]Main endpoint for intelligent prediction (`/predict`)[cite: 70, 71]. |
| **Grafana Dashboard** | 3000 | [cite\_start]Monitoring metrics (Latency, Requests, Errors)[cite: 101]. |
| **MLflow UI** | 5000 | [cite\_start]View runs and registered models[cite: 30, 87]. |

-----

### 🤖 AI Agent Functionality

[cite\_start]The Agent IA acts as the smart entry point[cite: 71, 111]:

  * [cite\_start]Receives customer tickets via API[cite: 112].
  * [cite\_start]**Scrubs PII** (removes sensitive data)[cite: 74, 113].
  * [cite\_start]Routes the request to the optimal model based on **confidence, text length, and language**[cite: 72, 115].
  * [cite\_start]Returns prediction and an explanation of the model choice[cite: 72, 120].

-----

Would you like the core steps for the **CI/CD GitHub Actions workflow** outlined next?
