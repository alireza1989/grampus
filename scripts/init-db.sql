-- Nexus database initialization
-- Runs automatically when the pgvector container starts for the first time.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
