"""
Sample resumes and job descriptions for the talent-matching pipeline demo.
Each resume/JD is realistic enough to produce meaningful embeddings and matches.
"""

RESUMES = [
    {
        "id": "r1",
        "name": "Priya Sharma",
        "text": """
Priya Sharma | priya.sharma@email.com | linkedin.com/in/priyasharma

SUMMARY
Machine Learning Engineer with 6 years of experience building production ML systems.
Passionate about NLP and large language models.

EXPERIENCE
Senior ML Engineer — DataSift AI (2021–present)
- Built a real-time document classification pipeline using BERT, serving 50M requests/day
- Fine-tuned LLaMA-2 for domain-specific QA; reduced hallucinations by 34%
- Led migration from Spark MLlib to PyTorch-based training on AWS SageMaker

ML Engineer — Flipkart (2019–2021)
- Developed recommendation engine using collaborative filtering + embeddings (AUC 0.91)
- Deployed FastAPI inference service with sub-30ms p99 latency

Junior Data Scientist — TCS (2018–2019)
- Built churn prediction model (XGBoost) for telecom client; 22% reduction in churn

SKILLS
Python, PyTorch, Hugging Face Transformers, LangChain, FastAPI, AWS SageMaker,
PostgreSQL, Redis, Docker, Kubernetes, SQL, PySpark

EDUCATION
B.Tech Computer Science — IIT Bombay, 2018
""",
    },
    {
        "id": "r2",
        "name": "Marcus Johnson",
        "text": """
Marcus Johnson | marcus.j@email.com | github.com/marcusj

SUMMARY
Full-Stack Engineer with 4 years of experience, recently pivoting into AI engineering.
Strong React/Node.js background; built two LLM-powered products in the last year.

EXPERIENCE
Software Engineer II — Stripe (2022–present)
- Built internal tooling dashboard (React + TypeScript + GraphQL) used by 300+ engineers
- Integrated OpenAI GPT-4 for automated dispute summarization, saving 8 hrs/day analyst time
- Designed webhook processing pipeline handling 2M events/day (Node.js + Kafka)

Software Engineer — Startup (Stealth) (2020–2022)
- Full-stack development: Next.js frontend, Express.js backend, PostgreSQL
- Implemented JWT auth, Stripe billing, and SendGrid email flows

SKILLS
TypeScript, React, Node.js, Next.js, GraphQL, PostgreSQL, Redis,
OpenAI API, LangChain (JS), Docker, GCP, Kafka

EDUCATION
B.S. Computer Science — University of Michigan, 2020
""",
    },
    {
        "id": "r3",
        "name": "Elena Vasquez",
        "text": """
Elena Vasquez | elena.v@email.com

SUMMARY
Data Engineer with 8 years building large-scale data infrastructure.
Expert in streaming pipelines and cloud-native data platforms.

EXPERIENCE
Staff Data Engineer — Snowflake (2020–present)
- Designed petabyte-scale ELT pipelines using dbt + Snowflake + Airflow
- Built real-time CDC pipeline with Debezium + Kafka + Iceberg; 0 data loss in 3 years
- Mentored team of 6 engineers; authored internal data quality framework

Senior Data Engineer — Uber (2017–2020)
- Owned pricing data pipelines (Flink + HDFS); critical path for surge pricing feature
- Led data platform migration from on-prem Hadoop to AWS EMR

Data Engineer — Oracle (2016–2017)
- ETL pipeline development with Oracle Data Integrator and PL/SQL

SKILLS
Python, SQL, dbt, Apache Kafka, Apache Flink, Apache Spark, Airflow,
Snowflake, AWS (EMR, Glue, S3), Terraform, Docker

EDUCATION
M.S. Information Systems — Carnegie Mellon, 2016
""",
    },
    {
        "id": "r4",
        "name": "James Park",
        "text": """
James Park | james.park@email.com | jamespark.dev

SUMMARY
AI Research Engineer with 3 years of experience, focused on RAG systems and
vector databases. Published 2 papers on retrieval-augmented generation.

EXPERIENCE
AI Engineer — Cohere (2023–present)
- Built production RAG pipeline using pgvector + LangChain serving enterprise clients
- Implemented hybrid search (BM25 + dense retrieval) improving recall@10 by 18%
- Developed evaluation framework for RAG quality (faithfulness, relevance, groundedness)

Research Engineer — Stanford NLP Lab (2021–2023)
- Co-authored "Adaptive Retrieval for Open-Domain QA" (ACL 2023)
- Implemented dense passage retrieval with FAISS and Sentence Transformers
- Ran large-scale experiments on NaturalQuestions and TriviaQA benchmarks

SKILLS
Python, PyTorch, Sentence Transformers, FAISS, pgvector, LangChain,
Hugging Face, PostgreSQL, Elasticsearch, BM25, Docker, Linux

EDUCATION
M.S. Computer Science (AI) — Stanford University, 2021
B.S. Mathematics — UC Berkeley, 2019
""",
    },
    {
        "id": "r5",
        "name": "Aisha Williams",
        "text": """
Aisha Williams | aisha.w@email.com

SUMMARY
Backend Engineer with 5 years specializing in API design and distributed systems.
Recently completed AWS ML Specialty certification; building GenAI features at current role.

EXPERIENCE
Senior Backend Engineer — Twilio (2021–present)
- Designed and owns the voice AI pipeline integrating Whisper STT + GPT-4 for IVR
- Built high-availability REST APIs (Python/FastAPI) handling 10M req/day
- Implemented rate limiting, circuit breaking, and observability with Datadog

Backend Engineer — Palantir (2019–2021)
- Built data ingestion microservices for government client; Python + gRPC + Postgres
- Optimized slow queries (up to 40x speedup) through indexing and query rewriting

SKILLS
Python, FastAPI, PostgreSQL, Redis, gRPC, Docker, Kubernetes,
AWS (Lambda, ECS, RDS), OpenAI API, Whisper, Datadog, Terraform

EDUCATION
B.S. Computer Engineering — Georgia Tech, 2019
AWS ML Specialty Certified (2024)
""",
    },
]

JOB_DESCRIPTIONS = [
    {
        "id": "jd1",
        "title": "Senior ML Engineer — LLM Platform",
        "text": """
We are building the next generation of AI infrastructure and are looking for a
Senior ML Engineer to join our LLM Platform team.

RESPONSIBILITIES
- Design and productionize LLM fine-tuning and inference pipelines
- Build scalable model serving infrastructure (low latency, high throughput)
- Collaborate with research team to ship new model capabilities to production
- Own reliability and performance SLOs for our model serving stack

REQUIREMENTS
- 5+ years of ML engineering experience
- Deep expertise in PyTorch and Hugging Face ecosystem
- Experience fine-tuning or deploying LLMs in production
- Strong Python skills; experience with FastAPI or similar
- Familiarity with cloud ML platforms (SageMaker, Vertex AI, or similar)
- Bonus: experience with RLHF, quantization, or vLLM

TECH STACK
Python, PyTorch, Hugging Face, AWS SageMaker, FastAPI, Kubernetes, Redis
""",
    },
    {
        "id": "jd2",
        "title": "AI Engineer — RAG & Search",
        "text": """
Join our Search & Discovery team to build state-of-the-art retrieval systems
that power our enterprise knowledge base product.

RESPONSIBILITIES
- Build and maintain hybrid search pipelines (semantic + keyword)
- Design vector storage and retrieval architecture using pgvector or similar
- Implement evaluation frameworks to measure and improve retrieval quality
- Integrate LLMs for query understanding, reranking, and answer generation

REQUIREMENTS
- 2+ years of experience with NLP or information retrieval
- Hands-on experience with vector databases (pgvector, Pinecone, FAISS, Weaviate)
- Strong Python skills; comfortable with Sentence Transformers or similar
- Familiarity with RAG architectures and retrieval evaluation metrics
- Experience with LangChain or LlamaIndex is a plus

TECH STACK
Python, pgvector, PostgreSQL, LangChain, Sentence Transformers, FastAPI, Docker
""",
    },
    {
        "id": "jd3",
        "title": "Staff Data Engineer — Streaming Platform",
        "text": """
We are scaling our real-time data platform to handle 100B events/day and need
a Staff Data Engineer to lead the effort.

RESPONSIBILITIES
- Architect and own our streaming ingestion platform (Kafka + Flink)
- Define data quality standards and build automated validation frameworks
- Drive adoption of modern data stack (dbt, Iceberg, Snowflake)
- Mentor junior and senior engineers; influence engineering roadmap

REQUIREMENTS
- 7+ years of data engineering experience
- Expert-level knowledge of Apache Kafka and stream processing (Flink or Spark Streaming)
- Strong experience with cloud data warehouses (Snowflake, BigQuery, or Redshift)
- Hands-on dbt experience; data modeling expertise
- Experience with infrastructure-as-code (Terraform, CDK)

TECH STACK
Python, SQL, Apache Kafka, Apache Flink, dbt, Snowflake, Airflow, Terraform, AWS
""",
    },
    {
        "id": "jd4",
        "title": "AI-Focused Full Stack Engineer",
        "text": """
We are a small team building AI-native B2B SaaS and need an engineer who can
own both the product UI and the AI backend integrations.

RESPONSIBILITIES
- Build responsive React/TypeScript frontend for our AI-powered product
- Integrate LLM APIs (OpenAI, Anthropic) into backend workflows
- Design and maintain PostgreSQL schema and API layer (Node.js or Python)
- Ship fast: take features from design to production in days, not weeks

REQUIREMENTS
- 3+ years of full-stack experience
- Strong TypeScript/React skills
- Experience integrating and prompting LLM APIs in production
- Comfortable with PostgreSQL and basic data modeling
- Startup mindset: self-directed, comfortable with ambiguity

TECH STACK
TypeScript, React, Next.js, Node.js, PostgreSQL, OpenAI API, LangChain, Docker
""",
    },
    {
        "id": "jd5",
        "title": "Senior Backend Engineer — AI Infrastructure",
        "text": """
Our AI Infrastructure team powers all of our product's AI features and we are
hiring a Senior Backend Engineer to help scale it.

RESPONSIBILITIES
- Build and maintain high-performance Python APIs for AI feature delivery
- Own integrations with third-party AI APIs (OpenAI, Whisper, ElevenLabs)
- Implement observability, alerting, and cost tracking for AI workloads
- Work with ML team to productionize new AI features end-to-end

REQUIREMENTS
- 4+ years of backend engineering experience
- Expert Python skills; FastAPI or similar async framework
- Production experience with OpenAI or other LLM APIs
- Strong PostgreSQL/Redis skills
- Experience with cloud infrastructure (AWS preferred); IaC with Terraform
- Bonus: speech/audio pipeline experience (Whisper, ASR)

TECH STACK
Python, FastAPI, PostgreSQL, Redis, AWS, OpenAI API, Docker, Kubernetes, Datadog
""",
    },
]
