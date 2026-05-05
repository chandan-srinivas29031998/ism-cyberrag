-- Enable extensions
create extension if not exists vector;

-- Documents table
create table documents (
  id bigserial primary key,
  title text not null,
  source_file text not null,
  content text,
  created_at timestamptz default now()
);

-- Chunks table with vector column and full-text search
create table chunks (
  id bigserial primary key,
  document_id bigint references documents(id),
  content text not null,
  control_id text,
  category text,
  sub_topic text,
  applicability text[],
  essential_8 text,
  revision int,
  embedding vector(768),
  fts tsvector generated always as (to_tsvector('english', content)) stored,
  created_at timestamptz default now()
);

-- Indexes
create index on chunks using hnsw (embedding vector_cosine_ops);
create index on chunks using gin (fts);

-- Row Level Security
-- Runtime app/notebook access is read-only via the Supabase publishable key.
-- No insert/update/delete policies are defined for anon/authenticated users.
alter table documents enable row level security;
alter table chunks enable row level security;

create policy "Allow read access to ISM documents"
on documents
for select
to anon, authenticated
using (true);

create policy "Allow read access to ISM chunks"
on chunks
for select
to anon, authenticated
using (true);

-- Sprint 1: Vector-only search (kept for backward compatibility)
create or replace function match_chunks(
  query_embedding vector(768),
  match_count int default 5
)
returns table (
  id bigint, content text, control_id text,
  category text, similarity float
)
language sql as $$
  select id, content, control_id, category,
         1 - (embedding <=> query_embedding) as similarity
  from chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- Sprint 2: Hybrid search combining vector similarity and BM25 full-text search
-- Uses Reciprocal Rank Fusion (RRF) to merge both ranked lists
create or replace function hybrid_search(
  query_text text,
  query_embedding vector(768),
  match_count int default 10,
  full_text_weight float default 1,
  semantic_weight float default 1,
  rrf_k int default 50
)
returns table (
  id bigint,
  content text,
  control_id text,
  category text,
  sub_topic text,
  applicability text[],
  essential_8 text,
  revision int,
  similarity float,
  rrf_score float
)
language sql as $$
  with full_text as (
    select c.id,
      row_number() over (order by ts_rank_cd(c.fts, websearch_to_tsquery(query_text)) desc) as rank_ix
    from chunks c
    where c.fts @@ websearch_to_tsquery(query_text)
    limit least(match_count, 30) * 2
  ),
  semantic as (
    select c.id,
      row_number() over (order by c.embedding <=> query_embedding) as rank_ix,
      1 - (c.embedding <=> query_embedding) as similarity
    from chunks c
    order by c.embedding <=> query_embedding
    limit least(match_count, 30) * 2
  )
  select
    c.id, c.content, c.control_id, c.category, c.sub_topic,
    c.applicability, c.essential_8, c.revision,
    coalesce(s.similarity, 0)::float as similarity,
    (coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight +
     coalesce(1.0 / (rrf_k + s.rank_ix), 0.0) * semantic_weight)::float as rrf_score
  from full_text ft
  full outer join semantic s on ft.id = s.id
  join chunks c on c.id = coalesce(ft.id, s.id)
  order by rrf_score desc
  limit least(match_count, 30);
$$;
