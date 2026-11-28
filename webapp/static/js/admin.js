// /admin: live view of analyses (running, waiting, recent), people online, the gallery and security.
// Polls the server every 2 s; only analysis jobs are shown (preparing, saving and loading are internal steps).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const REFRESH_MS = 2000;

const T = {
  vi: {
    title: "Quản trị", login_hint: "Nhập mật khẩu quản trị để xem hệ thống.", login: "Đăng nhập", back: "← Về trang tìm kiếm",
    open_app: "Mở trang tìm kiếm", logout: "Đăng xuất",
    tab_overview: "Tổng quan", tab_gallery: "Thư viện", tab_security: "Bảo mật",
    now: "Đang phân tích", live: "trực tiếp", offline: "mất kết nối", queue: "Hàng đợi", recent: "Đã phân tích gần đây",
    online: "Đang online", gallery: "Thư viện (Supabase)", blocked: "IP bị chặn", password: "Đổi mật khẩu quản trị",
    block: "Chặn", unblock: "Bỏ chặn", save: "Lưu", cancel: "Huỷ", del: "Xoá", reanalyse: "Phân tích lại",
    ip_ph: "IP", reason_ph: "Lý do", pw_cur: "Mật khẩu hiện tại", pw_new: "Mật khẩu mới (≥ 8 ký tự)", pw_new2: "Nhập lại mật khẩu mới",
    s_cpu: "CPU", s_ram: "RAM", s_running: "Đang phân tích", s_waiting: "Đang chờ", s_online: "Đang online", s_gallery: "Thư viện",
    none_running: "Không có video nào đang phân tích.", none_waiting: "Không có video nào đang chờ.",
    none_recent: "Chưa có lượt phân tích nào từ lúc server chạy.", none_online: "Không có ai trong 5 phút qua.",
    started: "bắt đầu {t}", elapsed: "đã chạy {d}", eta: "còn khoảng {d}", waited: "chờ {d}", ago: "{d} trước",
    people: "{n} người", took: "mất {d}", cancelled: "đã huỷ", failed: "lỗi: {e}", again: "phân tích lại", queued_of: "/ tối đa {n}",
    st_prepare: "Chuẩn bị video", st_detect: "Phát hiện và theo vết", st_crops: "Cắt ảnh từng người", st_stitch: "Nối người bị mất dấu",
    st_clip: "Mã hoá CLIP", st_encode: "Mã hoá video",
    do_upload: "Đang tải video lên", do_analyze: "Vừa bấm phân tích", do_search: "Đang tìm người", do_open: "Mở video trong thư viện",
    do_browse: "Đang xem trang", do_watch: "Đang xem video", do_results: "Đang xem kết quả", do_admin: "Trang quản trị",
    st_analysing: "đang phân tích", st_waiting: "đang chờ",
    h_name: "Tên", h_size: "Dung lượng", h_people: "Số người", h_uploader: "IP upload", h_date: "Ngày", h_opens: "Lượt mở", h_reason: "Lý do",
    no_gallery: "Chưa cấu hình Supabase (SUPABASE_URL): không có thư viện.", gallery_sum: "{n} video · {used} / {limit} MB",
    supa_down: "Supabase không truy cập được: {e}", none: "Không có",
    confirm_del: "Xoá video \"{n}\" khỏi thư viện? Không khôi phục được.", confirm_block: "Chặn IP {ip}? Việc phân tích của IP này sẽ bị huỷ.",
    confirm_cancel: "Huỷ việc phân tích \"{n}\"?", pw_mismatch: "Hai mật khẩu mới không giống nhau", pw_ok: "Đã đổi mật khẩu",
    done: "Xong", err: "Lỗi",
    shutdown: "Tắt server", shutdown_pending: "Máy sẽ tắt trong vòng 1 phút. Bật lại bằng GitHub → Actions → EC2 power → start.",
    confirm_shutdown: "Tắt máy EC2 ngay bây giờ? Mọi người đang dùng sẽ mất kết nối. Bật lại bằng GitHub → Actions → EC2 power → start.",
    confirm_shutdown_busy: "Đang có video được phân tích hoặc chờ, việc đó sẽ bị dừng. Vẫn tắt máy?",
    s_autostop: "Tự tắt", as_off: "không bật", as_busy: "đang có việc", as_left: "còn {d}", as_after: "sau {n} phút rảnh", as_now: "đang tắt…",
  },
  en: {
    title: "Admin", login_hint: "Enter the admin password to see the system.", login: "Log in", back: "← Back to search",
    open_app: "Open search page", logout: "Log out",
    tab_overview: "Overview", tab_gallery: "Gallery", tab_security: "Security",
    now: "Analysing now", live: "live", offline: "disconnected", queue: "Queue", recent: "Recent analyses",
    online: "Online", gallery: "Gallery (Supabase)", blocked: "Blocked IPs", password: "Change admin password",
    block: "Block", unblock: "Unblock", save: "Save", cancel: "Cancel", del: "Delete", reanalyse: "Analyse again",
    ip_ph: "IP", reason_ph: "Reason", pw_cur: "Current password", pw_new: "New password (≥ 8 characters)", pw_new2: "New password again",
    s_cpu: "CPU", s_ram: "RAM", s_running: "Analysing", s_waiting: "Waiting", s_online: "Online", s_gallery: "Gallery",
    none_running: "No video is being analysed.", none_waiting: "No video is waiting.",
    none_recent: "No analyses since the server started.", none_online: "Nobody in the last 5 minutes.",
    started: "started {t}", elapsed: "{d} elapsed", eta: "about {d} left", waited: "waiting {d}", ago: "{d} ago",
    people: "{n} people", took: "took {d}", cancelled: "cancelled", failed: "error: {e}", again: "re-analysis", queued_of: "/ max {n}",
    st_prepare: "Preparing video", st_detect: "Detecting and tracking", st_crops: "Cropping each person", st_stitch: "Re-joining lost people",
    st_clip: "CLIP encoding", st_encode: "Video encoding",
    do_upload: "Uploading a video", do_analyze: "Started an analysis", do_search: "Searching for someone", do_open: "Opened a gallery video",
    do_browse: "Browsing", do_watch: "Watching a video", do_results: "Viewing results", do_admin: "Admin page",
    st_analysing: "analysing", st_waiting: "waiting",
    h_name: "Name", h_size: "Size", h_people: "People", h_uploader: "Uploader IP", h_date: "Date", h_opens: "Opens", h_reason: "Reason",
    no_gallery: "Supabase is not configured (SUPABASE_URL): no gallery.", gallery_sum: "{n} videos · {used} / {limit} MB",
    supa_down: "Supabase unreachable: {e}", none: "None",
    confirm_del: "Delete \"{n}\" from the gallery? This cannot be undone.", confirm_block: "Block IP {ip}? Its analyses will be cancelled.",
    confirm_cancel: "Cancel the analysis of \"{n}\"?", pw_mismatch: "The two new passwords differ", pw_ok: "Password changed",
    done: "Done", err: "Error",
    shutdown: "Stop server", shutdown_pending: "The machine stops within a minute. Start it again with GitHub → Actions → EC2 power → start.",
    confirm_shutdown: "Stop the EC2 machine now? Everyone using it loses the connection. Start it again with GitHub → Actions → EC2 power → start.",
    confirm_shutdown_busy: "A video is being analysed or waiting; it will be stopped. Stop the machine anyway?",
    s_autostop: "Auto stop", as_off: "off", as_busy: "busy", as_left: "{d} left", as_after: "after {n} idle min", as_now: "stopping…",
  },
};
let lang = (() => { try { return localStorage.getItem("ps.lang") || "vi"; } catch (e) { return "vi"; } })();
const t = (k, v = {}) => Object.entries(v).reduce((s, [a, b]) => s.replaceAll(`{${a}}`, b), T[lang][k] ?? T.vi[k] ?? k);
function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
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

const mb = (b) => (b / 2 ** 20).toFixed(1);
const gb = (b) => (b / 2 ** 30).toFixed(1);
const dur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
  : s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${Math.max(0, Math.round(s))}s`);
const when = (ts) => {
  const d = new Date(typeof ts === "number" ? ts * 1000 : ts);
  return d.toDateString() === new Date().toDateString() ? d.toLocaleTimeString(lang) : d.toLocaleString(lang);
};
const table = (head, rows) =>
  `<thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.length ? rows.join("")
    : `<tr><td class="empty" colspan="${head.length}">${esc(t("none"))}</td></tr>`}</tbody>`;
const empty = (msg) => `<p class="adm-empty">${esc(msg)}</p>`;

// ------------------------------------------------------------------ overview (live)
let ov = null;
async function refresh() {
  try {
    ov = await api("GET", "/api/admin/overview");
  } catch (e) {
    document.querySelectorAll(".live").forEach((el) => { el.classList.add("off"); el.textContent = t("offline"); });
    return;
  }
  document.querySelectorAll(".live").forEach((el) => { el.classList.remove("off"); el.textContent = t("live"); });
  const analyses = ov.jobs.filter((j) => j.kind === "analyze");
  renderStats(analyses);
  renderNow(analyses);
  renderQueue(analyses);
  renderRecent(analyses);
  renderOnline(analyses);
  renderBlocked();
  $("updated").textContent = new Date().toLocaleTimeString(lang);
}

function setCount(tab, n) {
  const b = document.querySelector(`#tabs [data-tab="${tab}"]`);
  b.querySelector(".count")?.remove();
  if (n) b.insertAdjacentHTML("beforeend", `<span class="count">${n}</span>`);
}

function stat(label, value, frac = null, warn = false) {
  return `<div class="adm-stat${warn ? " warn" : ""}"><span>${esc(label)}</span><b>${esc(value)}</b>${
    frac != null ? `<div class="meter"><i style="width:${Math.min(100, frac * 100)}%"></i></div>` : ""}</div>`;
}

function renderStats(analyses) {
  const s = ov.system;
  const running = analyses.filter((j) => j.state === "running").length;
  const waiting = analyses.filter((j) => j.state === "queued").length;
  const parts = [
    stat(t("s_cpu"), `${s.cpu_pct.toFixed(0)}%`, s.cpu_pct / 100, s.cpu_pct > 90),
    stat(t("s_ram"), `${gb(s.ram_used)} / ${gb(s.ram_total)} GB`, s.ram_used / s.ram_total, s.ram_used / s.ram_total > 0.9),
    stat(t("s_running"), String(running)),
    stat(t("s_waiting"), `${waiting} ${t("queued_of", { n: ov.limits.max_queue })}`, waiting / ov.limits.max_queue, waiting >= ov.limits.max_queue),
    stat(t("s_online"), String(ov.online.length)),
  ];
  if (ov.gallery) {
    const g = ov.gallery;
    parts.push(g.error ? `<div class="adm-stat bad"><span>${esc(t("s_gallery"))}</span><small>${esc(t("supa_down", { e: g.error.slice(0, 60) }))}</small></div>`
      : stat(t("s_gallery"), `${mb(g.used_bytes)} / ${mb(g.limit_bytes)} MB`, g.used_bytes / g.limit_bytes, g.used_bytes / g.limit_bytes > 0.85));
  }
  const pw = ov.power || {};
  if (pw.shutdown_requested) parts.push(stat(t("s_autostop"), t("as_now"), 1, true));
  else if (pw.autostop_min > 0) {
    const left = pw.autostop_min * 60 - pw.idle_s;
    parts.push(`<div class="adm-stat"><span>${esc(t("s_autostop"))}</span><b>${esc(pw.busy ? t("as_busy") : t("as_left", { d: dur(Math.max(0, left)) }))}</b>`
      + `<small class="muted">${esc(t("as_after", { n: pw.autostop_min }))}</small></div>`);
  }
  $("shutdown-banner").hidden = !pw.shutdown_requested;
  $("shutdown-btn").hidden = !!pw.shutdown_requested;
  $("stats").innerHTML = parts.join("");
  $("sys-line").textContent = `· ${s.checkpoint || "–"} · ${s.device}`;
  setCount("overview", running + waiting);
  setCount("gallery", ov.gallery?.videos || 0);
  setCount("security", ov.blocked.length);
}

function renderNow(analyses) {
  const j = analyses.find((x) => x.state === "running");
  if (!j) { $("now").innerHTML = empty(t("none_running")); return; }
  $("now").innerHTML = `
    <div class="now">
      <div class="now-top"><b title="${esc(j.video_name)}">${esc(j.video_name || "?")}</b>${j.reanalyse ? `<span class="badge running">${esc(t("again"))}</span>` : ""}</div>
      <div class="now-sub"><span class="mono">${esc(j.ip || "")}</span> · ${esc(t("started", { t: when(j.started) }))}</div>
      <div class="now-bar"><i style="width:${j.pct}%"></i></div>
      <div class="now-meta">
        <span><b>${Math.round(j.pct)}%</b> · ${esc(T[lang][`st_${j.stage}`] || j.stage || "")}</span>
        <span class="muted">${esc(t("elapsed", { d: dur(j.elapsed_s) }))}${j.eta_s != null ? ` · ${esc(t("eta", { d: dur(j.eta_s) }))}` : ""}</span>
        <button class="link-danger" data-cancel="${esc(j.id)}" data-name="${esc(j.video_name)}">${esc(t("cancel"))}</button>
      </div>
    </div>`;
}

function renderQueue(analyses) {
  const q = analyses.filter((j) => j.state === "queued").sort((a, b) => a.created - b.created);
  const now = Date.now() / 1000;
  $("queue").innerHTML = q.length ? q.map((j, i) => `
    <div class="row">
      <span class="pos">#${i + 1}</span>
      <div class="row-main"><b title="${esc(j.video_name)}">${esc(j.video_name || "?")}</b>
        <small><span class="mono">${esc(j.ip || "")}</span> · ${esc(t("waited", { d: dur(now - j.created) }))}${j.reanalyse ? ` · ${esc(t("again"))}` : ""}</small></div>
      <button class="link-danger" data-cancel="${esc(j.id)}" data-name="${esc(j.video_name)}">${esc(t("cancel"))}</button>
    </div>`).join("") : empty(t("none_waiting"));
}

function renderRecent(analyses) {
  const done = analyses.filter((j) => ["done", "error", "cancelled"].includes(j.state))
    .sort((a, b) => (b.finished || b.created) - (a.finished || a.created)).slice(0, 15);
  $("recent").innerHTML = done.length ? done.map((j) => {
    const icon = j.state === "done" ? "✓" : j.state === "cancelled" ? "✕" : "⚠";
    const what = j.state === "done"
      ? [j.tracks != null ? t("people", { n: j.tracks }) : "", t("took", { d: dur(j.elapsed_s) })].filter(Boolean).join(" · ")
      : j.state === "cancelled" ? t("cancelled") : t("failed", { e: j.error || "?" });
    return `
    <div class="row ${j.state}">
      <span class="icon">${icon}</span>
      <div class="row-main"><b title="${esc(j.video_name)}">${esc(j.video_name || "?")}</b> <span class="what">${esc(what)}</span>
        <small><span class="mono">${esc(j.ip || "")}</span> · ${esc(when(j.finished || j.created))}${j.reanalyse ? ` · ${esc(t("again"))}` : ""}</small></div>
    </div>`;
  }).join("") : empty(t("none_recent"));
}

function doing(o) {
  const m = o.last_method, p = o.last_path || "";
  if (m === "POST" && p === "/api/videos") return t("do_upload");
  if (p.endsWith("/analyze")) return t("do_analyze");
  if (p.endsWith("/search")) return t("do_search");
  if (p.startsWith("/api/gallery") && p.endsWith("/open")) return t("do_open");
  if (p.endsWith("/stream")) return t("do_watch");
  if (p.startsWith("/api/analyses/")) return t("do_results");
  if (p === "/admin") return t("do_admin");
  return t("do_browse");
}

function renderOnline(analyses) {
  const now = Date.now() / 1000;
  const state = {};
  analyses.forEach((j) => {
    if (j.state === "running") state[j.ip] = "st_analysing";
    else if (j.state === "queued" && !state[j.ip]) state[j.ip] = "st_waiting";
  });
  $("online").innerHTML = ov.online.length ? ov.online.map((o) => `
    <div class="row">
      <div class="row-main"><b class="mono" title="${esc(o.user_agent || "")}">${esc(o.ip)}</b>${state[o.ip] ? ` <span class="badge running">${esc(t(state[o.ip]))}</span>` : ""}
        <small>${esc(doing(o))} · ${esc(t("ago", { d: dur(now - o.last_seen) }))}</small></div>
      <button class="link-danger" data-block="${esc(o.ip)}">${esc(t("block"))}</button>
    </div>`).join("") : empty(t("none_online"));
}

function renderBlocked() {
  $("blocked").innerHTML = table(["IP", t("h_reason"), ""], ov.blocked.map((b) => `<tr>
    <td class="mono">${esc(b.ip)}</td><td class="wrap">${esc(b.reason)}</td>
    <td><button class="link-act" data-unblock="${esc(b.ip)}">${esc(t("unblock"))}</button></td></tr>`));
}

// ------------------------------------------------------------------ gallery (on demand)
async function loadGallery() {
  let g;
  try { g = await api("GET", "/api/admin/gallery"); } catch (e) { return; }
  if (!g.enabled) { $("gallery-sum").textContent = t("no_gallery"); $("gallery").innerHTML = ""; return; }
  const rows = g.videos.map((v) => {
    const an = v.analyses || [];
    const size = v.proxy_bytes + (v.thumb_bytes || 0) + an.reduce((a, x) => a + x.blob_bytes, 0);
    return `<tr>
      <td><img src="/api/gallery/${esc(v.id)}/thumb" alt="" loading="lazy"></td>
      <td class="ell" title="${esc(v.name)}">${esc(v.name)}</td>
      <td class="num">${mb(size)} MB</td>
      <td class="num">${an.length ? an[0].n_tracks : "–"}</td>
      <td class="mono">${esc(v.uploader_ip || "")}</td>
      <td class="num">${esc(when(v.created_at))}</td>
      <td class="num">${v.opens ?? 0}</td>
      <td><button class="link-act" data-reanalyse="${esc(v.id)}">${esc(t("reanalyse"))}</button>
          <button class="link-danger" data-delete="${esc(v.id)}" data-name="${esc(v.name)}">${esc(t("del"))}</button></td></tr>`;
  });
  $("gallery-sum").textContent = g.error ? t("supa_down", { e: g.error })
    : ov?.gallery ? t("gallery_sum", { n: g.videos.length, used: mb(ov.gallery.used_bytes), limit: mb(ov.gallery.limit_bytes) }) : "";
  $("gallery").innerHTML = table(["", t("h_name"), t("h_size"), t("h_people"), t("h_uploader"), t("h_date"), t("h_opens"), ""], rows);
}

// ------------------------------------------------------------------ actions
document.addEventListener("click", async (e) => {
  const b = e.target.closest("button");
  if (!b || !(b.dataset.cancel || b.dataset.block || b.dataset.unblock || b.dataset.delete || b.dataset.reanalyse)) return;
  try {
    if (b.dataset.cancel) {
      if (!confirm(t("confirm_cancel", { n: b.dataset.name || "" }))) return;
      await api("POST", `/api/admin/jobs/${b.dataset.cancel}/cancel`);
    } else if (b.dataset.block) {
      if (!confirm(t("confirm_block", { ip: b.dataset.block }))) return;
      await api("POST", "/api/admin/block", { ip: b.dataset.block, reason: "" });
    } else if (b.dataset.unblock) {
      await api("DELETE", `/api/admin/block/${encodeURIComponent(b.dataset.unblock)}`);
    } else if (b.dataset.delete) {
      if (!confirm(t("confirm_del", { n: b.dataset.name }))) return;
      await api("DELETE", `/api/admin/gallery/${b.dataset.delete}`);
      toast("success", t("done"));
      loadGallery();
    } else if (b.dataset.reanalyse) {
      b.disabled = true;
      await api("POST", `/api/admin/gallery/${b.dataset.reanalyse}/reanalyse`);
      toast("success", t("done"));
    }
    refresh();
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

document.querySelectorAll("#tabs [data-tab]").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#tabs [data-tab]").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".adm-view").forEach((v) => { v.hidden = v.dataset.view !== b.dataset.tab; });
  if (b.dataset.tab === "gallery") loadGallery();
}));

$("lang-btn").addEventListener("click", () => {
  lang = lang === "vi" ? "en" : "vi";
  try { localStorage.setItem("ps.lang", lang); } catch (e) {}
  applyLang();
  refresh();
  loadGallery();
});

$("shutdown-btn").addEventListener("click", async () => {
  const busy = ov?.jobs.some((j) => j.kind === "analyze" && (j.state === "running" || j.state === "queued"));
  if (!confirm(t("confirm_shutdown")) || (busy && !confirm(t("confirm_shutdown_busy")))) return;
  try { await api("POST", "/api/admin/shutdown"); refresh(); } catch (err) { if (err.message !== "login") toast("error", err.message); }
});
$("shutdown-cancel").addEventListener("click", async () => {
  try { await api("POST", "/api/admin/shutdown/cancel"); refresh(); } catch (err) { if (err.message !== "login") toast("error", err.message); }
});

$("logout").addEventListener("click", async () => { await api("POST", "/api/admin/logout").catch(() => {}); showLogin(); });

// ------------------------------------------------------------------ login and start
let timer = null;
function showLogin() {
  clearInterval(timer);
  $("dash").hidden = true;
  $("login").hidden = false;
  $("login-pw").focus();
}

function showDash() {
  $("login").hidden = true;
  $("dash").hidden = false;
  refresh().then(loadGallery);
  clearInterval(timer);
  timer = setInterval(refresh, REFRESH_MS);
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
