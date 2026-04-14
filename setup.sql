-- Run once to set up the database.
-- Prerequisites: PostgreSQL 14+ with pgvector extension installed.
--
-- Install pgvector:  https://github.com/pgvector/pgvector
-- Quick Docker run:
--   docker run -e POSTGRES_PASSWORD=password -p 5432:5432 \
--              ankane/pgvector:latest
--
-- Then: psql -U postgres -f setup.sql

-- Create the database (run this outside a transaction if needed)
-- CREATE DATABASE talent_match;
-- \c talent_match

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Candidates table
CREATE TABLE IF NOT EXISTS candidates (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    raw_resume   TEXT NOT NULL,
    profile_json JSONB NOT NULL,
    embedding    vector(384)            -- all-MiniLM-L6-v2 output dimension
);

-- IVFFlat index for cosine similarity search.
-- For small datasets (< 1000 rows) an exact scan is fine — this index
-- becomes important at scale. lists ~= sqrt(num_rows) is a good heuristic.
CREATE INDEX IF NOT EXISTS candidates_embedding_idx
    ON candidates
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
