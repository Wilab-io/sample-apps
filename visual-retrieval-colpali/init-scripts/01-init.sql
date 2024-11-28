CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE ranker_type AS ENUM ('colpali', 'bm25', 'hybrid-colpali-bm25');

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
    FOREIGN KEY (user_id) REFERENCES app_user(user_id)
);

CREATE TABLE user_settings (
    user_id UUID PRIMARY KEY,
    demo_questions TEXT[] DEFAULT ARRAY[]::TEXT[],
    ranker ranker_type NOT NULL DEFAULT 'colpali',
    vespa_host VARCHAR(255),
    vespa_port INTEGER,
    vespa_token VARCHAR(255),
    gemini_token VARCHAR(255),
    vespa_cloud_endpoint VARCHAR(255),
    schema VARCHAR(255),
    prompt TEXT DEFAULT 'You are an investor, stock analyst and financial expert. You will be presented an image of a document page from a report published by the Norwegian Government Pension Fund Global (GPFG). The report may be annual or quarterly reports, or policy reports, on topics such as responsible investment, risk etc.
Your task is to generate retrieval queries and questions that you would use to retrieve this document (or ask based on this document) in a large corpus.
Please generate 3 different types of retrieval queries and questions.
A retrieval query is a keyword based query, made up of 2-5 words, that you would type into a search engine to find this document.
A question is a natural language question that you would ask, for which the document contains the answer.
The queries should be of the following types:
1. A broad topical query: This should cover the main subject of the document.
2. A specific detail query: This should cover a specific detail or aspect of the document.
3. A visual element query: This should cover a visual element of the document, such as a chart, graph, or image.

Important guidelines:
- Ensure the queries are relevant for retrieval tasks, not just describing the page content.
- Use a fact-based natural language style for the questions.
- Frame the queries as if someone is searching for this document in a large corpus.
- Make the queries diverse and representative of different search strategies.

Format your response as a JSON object with the structure of the following example:
{
    "broad_topical_question": "What was the Responsible Investment Policy in 2019?",
    "broad_topical_query": "responsible investment policy 2019",
    "specific_detail_question": "What is the percentage of investments in renewable energy?",
    "specific_detail_query": "renewable energy investments percentage",
    "visual_element_question": "What is the trend of total holding value over time?",
    "visual_element_query": "total holding value trend"
}

If there are no relevant visual elements, provide an empty string for the visual element question and query.
Here is the document image to analyze:
Generate the queries based on this image and provide the response in the specified JSON format.
Only return JSON. Don''t return any extra explanation text.',
    FOREIGN KEY (user_id) REFERENCES app_user(user_id)
);

CREATE INDEX idx_user_document_user_id ON user_document(user_id);
