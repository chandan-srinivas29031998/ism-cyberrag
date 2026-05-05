-- Sprint 3 RLS hardening
-- Run this against an existing Sprint 2 database.
--
-- The Sprint 3 app and notebook use SUPABASE_PUBLISHABLE_KEY and only read
-- the existing ISM corpus. These policies allow read access for retrieval
-- while leaving insert/update/delete unavailable to anon/authenticated users.

begin;

alter table public.chunks enable row level security;
alter table public.documents enable row level security;

drop policy if exists "Allow read access to ISM chunks" on public.chunks;
drop policy if exists "Allow read access to ISM documents" on public.documents;

create policy "Allow read access to ISM chunks"
on public.chunks
for select
to anon, authenticated
using (true);

create policy "Allow read access to ISM documents"
on public.documents
for select
to anon, authenticated
using (true);

commit;
