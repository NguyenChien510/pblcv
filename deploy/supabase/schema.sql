-- Web demo gallery (webapp/store.py) and /admin data (webapp/monitor.py, webapp/admin.py).
-- Run once in Supabase: Dashboard -> SQL Editor -> paste -> Run. Safe to run again.
-- Only the server touches these, with the project's secret key (which bypasses RLS). RLS is on with no policies,
-- so the publishable/anon key and the browser cannot read or write anything here.

create table if not exists public.videos (
  id            text primary key,             -- sha1 of the uploaded file (first 16 hex), as in the app
  name          text not null,
  proxy_bytes   bigint not null,               -- 720p H.264 proxy in Storage: videos/<id>/proxy.mp4 or proxy.partNNN
  proxy_parts   int not null default 0,        -- 0 = one object; n = n parts (Free plan: 50 MB per object)
  thumb_bytes   int not null default 0,        -- videos/<id>/thumb.jpg
  duration      double precision,
  fps           double precision,
  width         int,
  height        int,
  uploader_ip   text,
  opens         int not null default 0,
  created_at    timestamptz not null default now(),
  last_used_at  timestamptz not null default now()
);

create table if not exists public.analyses (
  id               text primary key,           -- analysis id; Storage: videos/<video_id>/<id>.zip
  video_id         text not null references public.videos(id) on delete cascade,
  params           jsonb not null,             -- max_frames, stride, conf, clip, tracker
  tracker_version  text not null,
  ckpt_key         text,
  n_tracks         int,
  blob_bytes       bigint not null,
  created_at       timestamptz not null default now()
);
create index if not exists analyses_video_idx on public.analyses (video_id);

-- no longer written since the admin page shows analyses live (2026-11-28); kept so existing projects need no change
create table if not exists public.activity_log (
  id        bigint generated always as identity primary key,
  ts        timestamptz not null default now(),
  ip        text,
  action    text not null,
  video_id  text,
  detail    jsonb
);
create index if not exists activity_ts_idx on public.activity_log (ts desc);
create index if not exists activity_ip_idx on public.activity_log (ip);

create table if not exists public.blocked_ips (
  ip          text primary key,
  reason      text,
  created_at  timestamptz not null default now()
);

create table if not exists public.app_settings (
  key         text primary key,                -- admin_password_hash
  value       text,
  updated_at  timestamptz not null default now()
);

alter table public.videos       enable row level security;
alter table public.analyses     enable row level security;
alter table public.activity_log enable row level security;
alter table public.blocked_ips  enable row level security;
alter table public.app_settings enable row level security;

-- Private bucket for proxies, thumbnails and analysis zips
insert into storage.buckets (id, name, public)
values ('videos', 'videos', false)
on conflict (id) do nothing;
