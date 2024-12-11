CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE ranker_type AS ENUM ('colpali', 'bm25', 'hybrid');

CREATE TABLE app_user (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE user_document (
    document_id VARCHAR(255) PRIMARY KEY,
    user_id UUID NOT NULL,
    document_name VARCHAR(255) NOT NULL,
    upload_ts TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    file_extension VARCHAR NOT NULL,
    FOREIGN KEY (user_id) REFERENCES app_user(user_id)
);

CREATE TABLE user_settings (
    user_id UUID PRIMARY KEY,
    demo_questions TEXT[] DEFAULT ARRAY[]::TEXT[],
    ranker ranker_type NOT NULL DEFAULT 'colpali',
    vespa_token_id VARCHAR(255),
    vespa_token_value VARCHAR(255),
    gemini_token VARCHAR(255),
    vespa_cloud_endpoint VARCHAR(255),
    tenant_name VARCHAR(255),
    app_name VARCHAR(255),
    instance_name VARCHAR(255),
    schema TEXT,
    prompt TEXT,
    FOREIGN KEY (user_id) REFERENCES app_user(user_id)
);

CREATE INDEX idx_user_document_user_id ON user_document(user_id);
