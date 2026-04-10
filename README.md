Healthcare AI Agents
A production-grade healthcare AI platform built for Oracle Health AI scale featuring RAG pipelines over clinical documents, multi-agent clinical decision support, biomarker-driven treatment recommendation, and enterprise FastAPI model serving.

Python License: MIT LangChain FastAPI FHIR

Platform Overview
Module	Files	Description
src/rag/	clinical_rag.py, fhir_indexer.py, embeddings.py	RAG pipeline over FHIR R4 records and clinical guidelines
src/agents/	clinical_agent.py, drug_interaction_tool.py, summarizer_agent.py	LangGraph multi-agent clinical decision support
src/recommender/	treatment_recommender.py, ranking_engine.py	Biomarker-driven treatment recommendation + ranking
src/serving/	api.py, Dockerfile, docker-compose.yml	FastAPI model serving, health checks, Docker deployment
Key Features
Clinical RAG Pipeline — Retrieval-augmented generation over FHIR R4 patient records and clinical guidelines using ChromaDB vector store and OpenAI embeddings
Multi-Agent CDS — LangGraph-orchestrated clinical decision support agents for drug interaction checking, contraindication alerts, and patient history summarization
Treatment Recommender — Biomarker-driven treatment ranking combining collaborative filtering, clinical rules, and patient profiles
Enterprise Serving — FastAPI REST API with JWT auth, rate limiting, health checks, model versioning, and Docker/OCI deployment
Tech Stack
Category	Libraries
LLM / RAG	openai, langchain, langchain-community, chromadb
Agents	langgraph, langchain-core, openai tools
ML	scikit-learn, TensorFlow, numpy, pandas
FHIR	fhir.resources, requests
Serving	FastAPI, uvicorn, pydantic, docker
Testing	pytest, pytest-asyncio, httpx
Getting Started
git clone https://github.com/rajeev19g-ship-it/healthcare-ai-agents.git
cd healthcare-ai-agents
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
uvicorn src.serving.api:app --reload
Architecture
FHIR R4 Records / Clinical Guidelines │ ▼ ┌─────────────────────────┐ │ RAG Pipeline │ Embed → Index → Retrieve → Generate └─────────────────────────┘ │ ▼ ┌─────────────────────────┐ │ AI Agent Layer │ Drug checks · Summarization · CDS └─────────────────────────┘ │ ▼ ┌─────────────────────────┐ │ Recommendation Engine │ Treatment ranking · Patient stratification └─────────────────────────┘ │ ▼ ┌─────────────────────────┐ │ FastAPI Serving │ REST API · Auth · Docker · OCI-ready └─────────────────────────┘

Regulatory & Clinical Standards
HL7 FHIR R4 — Patient data interoperability
SNOMED CT — Clinical terminology
RxNorm — Drug terminology
ICD-10-CM — Diagnosis coding
FDA Drug Interaction Guidance
Author
Girish Rajeev Clinical Data Scientist | Data Analyst | Regulatory Standards Leader | AI/ML Solution Engineer
