// /admin: system, jobs and queue, people online, blocked IPs, gallery and activity log. Polls the server every 3 s.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const T = {
  vi: {
    tab_overview: "Tổng quan", tab_gallery: "Thư viện", tab_activity: "Nhật ký", tab_security: "Bảo mật",
    title: "Quản trị", login_hint: "Nhập mật khẩu quản trị để xem hệ thống.", login: "Đăng nhập", back: "← Về trang tìm kiếm",
    open_app: "Mở trang tìm kiếm", logout: "Đăng xuất", jobs: "Job và hàng đợi", online: "Đang online (5 phút)",
    blocked: "IP bị chặn", block: "Chặn", unblock: "Bỏ chặn", gallery: "Thư viện (Supabase)", activity: "Nhật ký hoạt động",
    all_actions: "Mọi sự kiện", src_memory: "Từ lúc server chạy", src_db: "Lịch sử (Supabase, 30 ngày)",
    password: "Đổi mật khẩu quản trị", save: "Lưu", ip_ph: "IP", reason_ph: "Lý do", ip_filter: "Lọc theo IP",
    pw_cur: "Mật khẩu hiện tại", pw_new: "Mật khẩu mới (≥ 8 ký tự)", pw_new2: "Nhập lại mật khẩu mới",
    cpu: "CPU", ram: "RAM", proc: "Tiến trình app", disk: "Đĩa còn trống", uptime: "Đã chạy", storage: "Thư viện Supabase",
    cache: "Cache trên đĩa", model: "Checkpoint", queue: "Hàng đợi", running: "chạy", waiting: "chờ",
    h_state: "Trạng thái", h_kind: "Loại", h_video: "Video", h_ip: "IP", h_progress: "Tiến độ", h_time: "Thời gian",
    h_created: "Gửi lúc", h_req: "Lượt", h_seen: "Lần cuối", h_path: "Đang làm", h_ua: "Trình duyệt", h_reason: "Lý do",
    h_name: "Tên", h_size: "Dung lượng", h_people: "Số người", h_uploader: "IP upload", h_date: "Ngày", h_opens: "Lượt mở",
    h_ts: "Lúc", h_action: "Sự kiện", h_detail: "Chi tiết", cancel: "Huỷ", del: "Xoá", reanalyse: "Phân tích lại",
    none: "Không có", no_gallery: "Chưa cấu hình Supabase (SUPABASE_URL): không có thư viện.", ago: "{s} giây",
    confirm_del: "Xoá video \"{n}\" khỏi thư viện? Không khôi phục được.", confirm_block: "Chặn IP {ip}? Job của IP này sẽ bị huỷ.",
    pw_mismatch: "Hai mật khẩu mới không giống nhau", pw_ok: "Đã đổi mật khẩu", done: "Xong", err: "Lỗi",
    supa_down: "Supabase không truy cập được: {e}", gallery_sum: "{n} video · {used} / {limit} MB", locked: "Sai quá nhiều lần, thử lại sau 10 phút",
    kind_prepare: "Chuẩn bị video", kind_analyze: "Phân tích", kind_fetch: "Tải từ thư viện",
    kind_load: "Nạp kết quả", kind_save_video: "Lưu video", kind_save_analysis: "Lưu kết quả",
    st_queued: "chờ", st_running: "chạy", st_done: "xong", st_error: "lỗi", st_cancelled: "huỷ",
  },
  en: {
    tab_overview: "Overview", tab_gallery: "Gallery", tab_activity: "Activity", tab_security: "Security",
    title: "Admin", login_hint: "Enter the admin password to see the system.", login: "Log in", back: "← Back to search",
    open_app: "Open search page", logout: "Log out", jobs: "Jobs and queue", online: "Online (5 minutes)",
    blocked: "Blocked IPs", block: "Block", unblock: "Unblock", gallery: "Gallery (Supabase)", activity: "Activity log",
    all_actions: "All events", src_memory: "Since the server started", src_db: "History (Supabase, 30 days)",
    password: "Change admin password", save: "Save", ip_ph: "IP", reason_ph: "Reason", ip_filter: "Filter by IP",
    pw_cur: "Current password", pw_new: "New password (≥ 8 characters)", pw_new2: "New password again",
    cpu: "CPU", ram: "RAM", proc: "App process", disk: "Free disk", uptime: "Uptime", storage: "Supabase gallery",
    cache: "Disk cache", model: "Checkpoint", queue: "Queue", running: "running", waiting: "waiting",
    h_state: "State", h_kind: "Kind", h_video: "Video", h_ip: "IP", h_progress: "Progress", h_time: "Time",
    h_created: "Submitted", h_req: "Hits", h_seen: "Last seen", h_path: "Doing", h_ua: "Browser", h_reason: "Reason",
    h_name: "Name", h_size: "Size", h_people: "People", h_uploader: "Uploader IP", h_date: "Date", h_opens: "Opens",
    h_ts: "Time", h_action: "Event", h_detail: "Detail", cancel: "Cancel", del: "Delete", reanalyse: "Analyse again",
    none: "None", no_gallery: "Supabase is not configured (SUPABASE_URL): no gallery.", ago: "{s}s",
    confirm_del: "Delete \"{n}\" from the gallery? This cannot be undone.", confirm_block: "Block IP {ip}? Its jobs will be cancelled.",
    pw_mismatch: "The two new passwords differ", pw_ok: "Password changed", done: "Done", err: "Error",
    supa_down: "Supabase unreachable: {e}", gallery_sum: "{n} videos · {used} / {limit} MB", locked: "Too many attempts, try again in 10 minutes",
    kind_prepare: "Prepare video", kind_analyze: "Analysis", kind_fetch: "Download from gallery",
    kind_load: "Load result", kind_save_video: "Save video", kind_save_analysis: "Save result",
    st_queued: "queued", st_running: "running", st_done: "done", st_error: "error", st_cancelled: "cancelled",
  },
};
let lang = (() => { try { return localStorage.getItem("ps.lang") || "vi"; } catch (e) { return "vi"; } })();
const t = (k, v = {}) => Object.entries(v).reduce((s, [a, b]) => s.replaceAll(`{${a}}`, b), T[lang][k] ?? T.vi[k] ?? k);
function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
  if (ov) renderStats();          // tab labels were reset: restore their counts
  document.querySelectorAll("[data-tp]").forEach((el) => { el.placeholder = t(el.dataset.tp); });
  $("lang-btn").textContent = lang === "vi" ? "EN" : "VI";
}

function toast(kind, msg) {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<div><b>${esc(msg)}</b></div>`;
  $("toasts").appendChild(el);
  setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 200); }, 4000);
}

async function api(method, url, body) {
  const r = await fetch(url, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
  let data = null;
  try { data = await r.json(); } catch (e) {}
  if (r.status === 401 && url !== "/api/admin/login") { showLogin(); throw new Error("login"); }
  if (!r.ok) throw new Error(data?.detail || `HTTP ${r.status}`);
  return data;
}

const mb = (b) => (b / 2 ** 20).toFixed(b >= 2 ** 30 ? 0 : 1);
const gb = (b) => (b / 2 ** 30).toFixed(1);
const dur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Math.round(s)}s`);
const clock = (ts) => new Date(typeof ts === "number" ? ts * 1000 : ts).toLocaleString(lang);
const table = (head, rows, empty = t("none")) =>
  `<thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.length ? rows.join("") : `<tr><td class="empty" colspan="${head.length}">${esc(empty)}</td></tr>`}</tbody>`;

// ------------------------------------------------------------------ overview (every 3 s)
let ov = null;
async function refresh() {
  try {
    ov = await api("GET", "/api/admin/overview");
  } catch (e) { return; }
  renderStats();
  renderJobs();
  renderOnline();
  $("updated").textContent = new Date().toLocaleTimeString(lang);
}

function stat(label, value, frac = null, cls = "") {
  return `<div class="adm-stat ${cls}"><span>${esc(label)}</span><b>${esc(value)}</b>${frac != null ? `<div class="meter"><i style="width:${Math.min(100, frac * 100)}%"></i></div>` : ""}</div>`;
}

function setCount(tab, n) {
  const b = document.querySelector(`#tabs [data-tab="${tab}"]`);
  b.querySelector(".count")?.remove();
  if (n) b.insertAdjacentHTML("beforeend", `<span class="count">${n}</span>`);
}

function renderStats() {
  const s = ov.system;
  const active = ov.jobs.filter((j) => j.state === "running" || j.state === "queued");
  setCount("overview", active.length);
  setCount("gallery", ov.gallery?.videos || 0);
  setCount("security", ov.blocked.length);
  const parts = [
    stat(t("cpu"), `${s.cpu_pct.toFixed(0)}% · ${s.cpus} vCPU`, s.cpu_pct / 100, s.cpu_pct > 90 ? "warn" : ""),
    stat(t("ram"), `${gb(s.ram_used)} / ${gb(s.ram_total)} GB`, s.ram_used / s.ram_total, s.ram_used / s.ram_total > 0.9 ? "warn" : ""),
    stat(t("proc"), `${gb(s.proc_rss)} GB`),
    stat(t("disk"), `${gb(s.disk_free)} / ${gb(s.disk_total)} GB`, 1 - s.disk_free / s.disk_total, s.disk_free < 5 * 2 ** 30 ? "warn" : ""),
    stat(t("queue"), `${active.filter((j) => j.state === "running").length} ${t("running")} · ${active.filter((j) => j.state === "queued").length} ${t("waiting")}`,
      active.length / ov.limits.max_queue, active.length >= ov.limits.max_queue ? "warn" : ""),
  ];
  $("sys-line").textContent = `· ${s.checkpoint || "–"} · ${s.device} · ${t("uptime")} ${dur(s.uptime_s)}`;
  if (ov.gallery) {
    const g = ov.gallery;
    parts.push(g.error ? stat(t("storage"), t("supa_down", { e: g.error.slice(0, 80) }), null, "bad")
      : stat(t("storage"), `${mb(g.used_bytes)} / ${mb(g.limit_bytes)} MB`, g.used_bytes / g.limit_bytes, g.used_bytes / g.limit_bytes > 0.85 ? "warn" : ""));
    parts.push(stat(t("cache"), ov.cache_bytes >= 2 ** 30 ? `${gb(ov.cache_bytes)} GB` : `${mb(ov.cache_bytes)} MB`));
  }
  $("stats").innerHTML = parts.join("");
}

function renderJobs() {
  const rows = ov.jobs.map((j) => {
    const live = j.state === "running" || j.state === "queued";
    const time = j.state === "running" ? `${dur(j.elapsed_s)}${j.eta_s != null ? ` · ETA ${dur(j.eta_s)}` : ""}` : j.started ? dur(j.elapsed_s) : "";
    return `<tr>
      <td><span class="badge ${j.state}">${esc(t(`st_${j.state}`))}</span></td>
      <td>${esc(t(`kind_${j.kind}`))}${j.reanalyse ? " ↻" : ""}</td>
      <td class="ell" title="${esc(j.error || j.video_name || "")}">${esc(j.video_name || "")}${j.error ? ` <span class="adm-err">⚠ ${esc(j.error)}</span>` : ""}</td>
      <td class="mono">${esc(j.ip || "")}</td>
      <td class="num">${j.state === "running" ? `<span class="pbar"><i style="width:${j.pct}%"></i></span>${j.pct}% ${esc(j.stage)}` : ""}</td>
      <td class="num">${esc(time)}</td>
      <td class="num">${esc(clock(j.created))}</td>
      <td>${live ? `<button class="link-danger" data-cancel="${esc(j.id)}">${esc(t("cancel"))}</button>` : ""}</td></tr>`;
  });
  $("jobs").innerHTML = table([t("h_state"), t("h_kind"), t("h_video"), t("h_ip"), t("h_progress"), t("h_time"), t("h_created"), ""], rows);
}

function renderOnline() {
  const now = Date.now() / 1000;
  const working = new Set(ov.jobs.filter((j) => j.state === "running" || j.state === "queued").map((j) => j.ip));
  const rows = ov.online.map((o) => `<tr>
    <td class="mono" title="${esc(o.user_agent || "")}">${esc(o.ip)}${working.has(o.ip) ? ' <span class="badge running">job</span>' : ""}</td>
    <td class="num">${o.requests}</td>
    <td class="num">${esc(t("ago", { s: Math.round(now - o.last_seen) }))}</td>
    <td class="mono ell path" title="${esc(o.last_path || "")}">${esc(o.last_path || "")}</td>
    <td><button class="link-danger" data-block="${esc(o.ip)}">${esc(t("block"))}</button></td></tr>`);
  $("online").innerHTML = table([t("h_ip"), t("h_req"), t("h_seen"), t("h_path"), ""], rows);   // browser: hover the IP
  $("blocked").innerHTML = table([t("h_ip"), t("h_reason"), ""], ov.blocked.map((b) => `<tr>
    <td class="mono">${esc(b.ip)}</td><td class="wrap">${esc(b.reason)}</td>
    <td><button class="link-act" data-unblock="${esc(b.ip)}">${esc(t("unblock"))}</button></td></tr>`));
}

// ------------------------------------------------------------------ gallery (on load and after actions)
async function loadGallery() {
  let g;
  try { g = await api("GET", "/api/admin/gallery"); } catch (e) { return; }
  if (!g.enabled) { $("gallery-sum").textContent = t("no_gallery"); $("gallery").innerHTML = ""; return; }
  if (g.error) $("gallery-sum").textContent = t("supa_down", { e: g.error });
  const rows = g.videos.map((v) => {
    const an = v.analyses || [];
    const size = v.proxy_bytes + (v.thumb_bytes || 0) + an.reduce((a, x) => a + x.blob_bytes, 0);
    return `<tr>
      <td><img src="/api/gallery/${esc(v.id)}/thumb" alt="" loading="lazy"></td>
      <td class="ell" title="${esc(v.name)}">${esc(v.name)}</td>
      <td class="num">${mb(size)} MB</td>
      <td class="num">${an.length ? an[0].n_tracks : "–"}</td>
      <td class="mono">${esc(v.uploader_ip || "")}</td>
      <td class="num">${esc(clock(v.created_at))}</td>
      <td class="num">${v.opens ?? 0}</td>
      <td><button class="link-act" data-reanalyse="${esc(v.id)}">${esc(t("reanalyse"))}</button>
          <button class="link-danger" data-delete="${esc(v.id)}" data-name="${esc(v.name)}">${esc(t("del"))}</button></td></tr>`;
  });
  if (!g.error && ov?.gallery) $("gallery-sum").textContent = t("gallery_sum", { n: g.videos.length, used: mb(ov.gallery.used_bytes), limit: mb(ov.gallery.limit_bytes) });
  $("gallery").innerHTML = table(["", t("h_name"), t("h_size"), t("h_people"), t("h_uploader"), t("h_date"), t("h_opens"), ""], rows);
}

// ------------------------------------------------------------------ activity (every 5 s)
const ACTIONS = ["upload", "gallery_open", "analysis_start", "analysis_done", "analysis_loaded", "analysis_saved",
  "search", "gallery_save", "gallery_save_failed", "analysis_save_failed", "admin_login",
  "admin_login_failed", "admin_block", "admin_unblock", "admin_delete_video", "admin_cancel_job", "admin_password_changed"];
async function loadActivity() {
  const q = new URLSearchParams({ ip: $("act-ip").value.trim(), action: $("act-action").value, source: $("act-source").value, limit: "300" });
  let res;
  try { res = await api("GET", `/api/admin/activity?${q}`); } catch (e) { if (e.message !== "login") toast("error", e.message); return; }
  const rows = res.events.map((e) => {
    const d = e.detail || {};
    const detail = Object.entries(d).map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : v}`).join(" · ");
    return `<tr><td class="num">${esc(clock(e.ts))}</td><td class="mono">${esc(e.ip)}</td><td>${esc(e.action)}</td>
      <td class="mono">${esc(e.video_id || "")}</td><td class="wrap">${esc(detail)}</td></tr>`;
  });
  $("activity").innerHTML = table([t("h_ts"), t("h_ip"), t("h_action"), t("h_video"), t("h_detail")], rows);
}

// ------------------------------------------------------------------ actions
document.addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  try {
    if (b.dataset.cancel) {
      await api("POST", `/api/admin/jobs/${b.dataset.cancel}/cancel`);
      refresh();
    } else if (b.dataset.block) {
      if (!confirm(t("confirm_block", { ip: b.dataset.block }))) return;
      await api("POST", "/api/admin/block", { ip: b.dataset.block, reason: "" });
      refresh();
    } else if (b.dataset.unblock) {
      await api("DELETE", `/api/admin/block/${encodeURIComponent(b.dataset.unblock)}`);
      refresh();
    } else if (b.dataset.delete) {
      if (!confirm(t("confirm_del", { n: b.dataset.name }))) return;
      await api("DELETE", `/api/admin/gallery/${b.dataset.delete}`);
      toast("success", t("done"));
      loadGallery();
    } else if (b.dataset.reanalyse) {
      b.disabled = true;
      await api("POST", `/api/admin/gallery/${b.dataset.reanalyse}/reanalyse`);
      toast("success", t("done"));
      refresh();
    } else return;
  } catch (err) {
    if (err.message !== "login") toast("error", `${t("err")}: ${err.message}`);
  } finally {
    b.disabled = false;
  }
});

$("block-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    await api("POST", "/api/admin/block", { ip: $("block-ip").value.trim(), reason: $("block-reason").value.trim() });
    $("block-ip").value = $("block-reason").value = "";
    refresh();
  } catch (err) { if (err.message !== "login") toast("error", err.message); }
});

$("pw-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if ($("pw-new").value !== $("pw-new2").value) return toast("error", t("pw_mismatch"));
  try {
    await api("POST", "/api/admin/password", { current: $("pw-cur").value, new: $("pw-new").value });
    e.target.reset();
    toast("success", t("pw_ok"));
  } catch (err) { if (err.message !== "login") toast("error", err.message); }
});

// tabs: one view at a time, so nothing needs a long page scroll
document.querySelectorAll("#tabs [data-tab]").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs [data-tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".adm-view").forEach((v) => { v.hidden = v.dataset.view !== b.dataset.tab; });
  if (b.dataset.tab === "gallery") loadGallery();
  if (b.dataset.tab === "activity") loadActivity();
}));

$("act-form").addEventListener("input", () => loadActivity());
$("act-form").addEventListener("submit", (e) => e.preventDefault());
$("act-action").insertAdjacentHTML("beforeend", ACTIONS.map((a) => `<option value="${a}">${a}</option>`).join(""));

$("lang-btn").addEventListener("click", () => {
  lang = lang === "vi" ? "en" : "vi";
  try { localStorage.setItem("ps.lang", lang); } catch (e) {}
  applyLang();
  refresh(); loadGallery(); loadActivity();
});

$("logout").addEventListener("click", async () => { await api("POST", "/api/admin/logout").catch(() => {}); showLogin(); });

// ------------------------------------------------------------------ login and start
let timers = [];
function showLogin() {
  timers.forEach(clearInterval);
  timers = [];
  $("dash").hidden = true;
  $("login").hidden = false;
  $("login-pw").focus();
}

function showDash() {
  $("login").hidden = true;
  $("dash").hidden = false;
  refresh().then(loadGallery);
  loadActivity();
  timers = [setInterval(refresh, 3000), setInterval(() => { if ($("act-source").value === "memory") loadActivity(); }, 5000)];
}

$("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("login-err").hidden = true;
  try {
    await api("POST", "/api/admin/login", { password: $("login-pw").value });
    $("login-pw").value = "";
    showDash();
  } catch (err) {
    $("login-err").textContent = err.message;
    $("login-err").hidden = false;
  }
});

applyLang();
api("GET", "/api/admin/overview").then(showDash).catch(() => {});
