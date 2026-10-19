/* front-end (vanilla JS, no build step) */
const $ = (s, r = document) => r.querySelector(s);
const app = $('#app');

/* ------------------------------------------------------------------ helpers */
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
  return e;
}
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
};
const post = (url, body) => api(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.remove('hidden');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add('hidden'), 1800);
}
/* Dataset image paths are from 2017 and many are dead -> ask the server for TMDB's current path. */
let keyWarned = false;
const freshCache = new Map();      // id -> {poster, poster_large, backdrop} | null
const freshWaiting = new Map();    // id -> [resolve, ...]
let freshTimer = null;
function freshImages(id) {
  if (freshCache.has(id)) return Promise.resolve(freshCache.get(id));
  return new Promise(resolve => {
    if (freshWaiting.has(id)) freshWaiting.get(id).push(resolve); else freshWaiting.set(id, [resolve]);
    if (!freshTimer) freshTimer = setTimeout(flushFresh, 80);   // collect many requests into one
  });
}
async function postWithRetry(ids) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try { return await post('/api/posters', { ids }); }
    catch (e) { await new Promise(r => setTimeout(r, 700 * (attempt + 1))); }
  }
  return null;
}
async function flushFresh() {
  freshTimer = null;
  const waiting = [...freshWaiting.entries()]; freshWaiting.clear();
  for (let i = 0; i < waiting.length; i += 30) {
    const chunk = waiting.slice(i, i + 30);
    const res = await postWithRetry(chunk.map(([id]) => id));
    if (res && res.error && !keyWarned) {
      keyWarned = true;
      showKeyBanner(res.error === 'bad_key' ? 'Your TMDB API key was rejected. Check the key in your .env file.'
        : 'Add a free TMDB API key (see README) to load fresh posters and new movies.');
    }
    for (const [id, resolvers] of chunk) {
      const val = res && res.results ? (res.results[String(id)] || null) : null;
      if (val) freshCache.set(id, val);          // only remember successes, failures can be retried later
      resolvers.forEach(r => r(val));
    }
  }
}
function showKeyBanner(msg) {
  if (document.getElementById('keyBanner') || sessionStorage.getItem('keyBannerOff')) return;
  document.body.append(el('div', { id: 'keyBanner', class: 'banner' },
    el('span', { text: msg || 'Some posters are outdated. Add a free TMDB API key (see README) to load the latest ones.' }),
    el('button', { text: '✕', onclick: e => { try { sessionStorage.setItem('keyBannerOff', '1'); } catch { } e.target.parentNode.remove(); } })));
}
function testImage(url) {
  return new Promise(res => { const i = new Image(); i.onload = () => res(true); i.onerror = () => res(false); i.src = url; });
}
async function resolveImage(id, url, kind) {
  if (url && await testImage(url)) return url;
  const f = await freshImages(id);
  const alt = f && f[kind];
  return alt && await testImage(alt) ? alt : null;
}
function img(src, alt, id, fbClass = 'fallback') {
  const wrap = document.createDocumentFragment();
  const fb = el('div', { class: fbClass, text: alt });
  fb.style.display = 'none';
  const i = el('img', { src, alt, loading: 'lazy' });
  let tried = false;
  i.addEventListener('error', async () => {
    if (!tried && id) {
      tried = true;
      const f = await freshImages(id);
      if (f && f.poster) { i.src = f.poster; return; }
    }
    i.style.display = 'none'; fb.style.display = 'flex';
  });
  wrap.append(i, fb);
  return wrap;
}
/* hero / modal header: backdrop if it loads, else blurred poster + sharp poster */
async function paintHeader(node, m, posterClass, blurClass) {
  const back = m.backdrop && await resolveImage(m.id, m.backdrop, 'backdrop');
  if (back) { node.style.backgroundImage = `url(${back})`; return; }
  const p = await resolveImage(m.id, m.poster_large, 'poster_large');
  if (!p) return;
  node.prepend(el('div', { class: blurClass, style: `background-image:url(${p})` }));
  node.append(el('img', { class: posterClass, src: p, alt: m.title }));
}

const isUpcoming = m => !!m.release_date && new Date(m.release_date + 'T00:00:00') > new Date();
const fmtDate = d => new Date(d + 'T00:00:00').toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });

/* ---------------------------------------------------------------- My List */
const LIST_KEY = 'flimflix_list_v1';
let myList = [];
try { myList = JSON.parse(localStorage.getItem(LIST_KEY) || '[]'); } catch { myList = []; }
const inList = id => myList.includes(id);
function saveList() {
  try { localStorage.setItem(LIST_KEY, JSON.stringify(myList)); } catch { }
  const c = $('#listCount'); c.textContent = myList.length; c.classList.toggle('hidden', !myList.length);
}
function toggleList(movie) {
  const id = movie.id;
  if (inList(id)) { myList = myList.filter(x => x !== id); toast(`Removed “${movie.title}” from My List`); }
  else { myList.push(id); toast(`Added “${movie.title}” to My List`); }
  saveList();
  document.querySelectorAll(`[data-add="${id}"]`).forEach(b => syncAddBtn(b, id));
  if (currentView === 'home') refreshPersonalRows();
  if (currentView === 'mylist') showMyList();
}
function syncAddBtn(b, id) {
  b.classList.toggle('on', inList(id));
  if (b.classList.contains('add')) b.textContent = inList(id) ? '✓' : '+';
  else b.textContent = inList(id) ? '✓ In My List' : '+ My List';
}

/* ------------------------------------------------------------------ cards */
function movieCard(m, showMatch = false) {
  const add = el('button', {
    class: 'add', 'data-add': m.id, title: 'My List', 'aria-label': 'Toggle My List',
    onclick: e => { e.stopPropagation(); toggleList(m); }
  });
  syncAddBtn(add, m.id);
  const meta = el('div', { class: 'm' },
    showMatch && m.match ? el('span', { class: 'match', text: `${m.match}% Match` }) : null,
    m.rating ? el('span', {}, el('span', { class: 'star', text: '★ ' }), m.rating.toFixed(1)) : null,
    isUpcoming(m) ? el('span', { text: `Releases ${fmtDate(m.release_date)}` }) : (m.year ? el('span', { text: m.year }) : null));
  const ov = el('div', { class: 'ov' }, el('div', { class: 't', text: m.title }), meta,
    el('div', { class: 'm', style: 'margin-top:3px;color:#aaa' }, (m.genres || []).slice(0, 2).join(' • ')));
  const c = el('div', {
    class: 'card', tabindex: '0', onclick: () => openMovie(m.id),
    onkeydown: e => { if (e.key === 'Enter') openMovie(m.id); }
  }, img(m.poster, m.title, m.id), ov, add,
    isUpcoming(m) ? el('div', { class: 'ribbon', text: `Coming ${new Date(m.release_date + 'T00:00:00').toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}` })
      : (m.live && m.year >= new Date().getFullYear() - 0 ? el('div', { class: 'ribbon new', text: 'NEW' }) : null));
  return c;
}

function rowEl(title, movies, { showMatch = false, sub = '' } = {}) {
  const sc = el('div', { class: 'scroller' }, movies.map(m => movieCard(m, showMatch)));
  const go = dir => sc.scrollBy({ left: dir * sc.clientWidth * 0.9, behavior: 'smooth' });
  return el('section', { class: 'row' },
    el('h2', { text: title }, sub ? el('small', { text: sub }) : null),
    el('button', { class: 'arrow l', 'aria-label': 'Scroll left', text: '‹', onclick: () => go(-1) }),
    sc,
    el('button', { class: 'arrow r', 'aria-label': 'Scroll right', text: '›', onclick: () => go(1) }));
}

/* ------------------------------------------------------------------- views */
let currentView = 'home';
function setActive(name) {
  currentView = name;
  document.querySelectorAll('.links a[data-nav]').forEach(a => a.classList.toggle('active', a.dataset.nav === name));
}
const yt = m => `https://www.youtube.com/results?search_query=${encodeURIComponent(`${m.title} ${m.year || ''} official trailer`)}`;

function heroEl(h) {
  const hero = el('section', { class: 'hero' });
  paintHeader(hero, h, 'hero-poster', 'hero-blur');
  const addBtn = el('button', { class: 'btn btn-grey', 'data-add': h.id, onclick: () => toggleList(h) });
  syncAddBtn(addBtn, h.id);
  hero.append(el('div', { class: 'hero-info' },
    el('div', { class: 'hero-tag', text: h.collection ? h.collection : 'Featured' }),
    el('h1', { class: 'hero-title', text: h.title }),
    el('div', { class: 'hero-meta' },
      el('span', { class: 'match', text: `★ ${h.rating.toFixed(1)}` }),
      h.year ? el('span', { text: h.year }) : null,
      h.runtime ? el('span', { class: 'badge', text: `${Math.floor(h.runtime / 60)}h ${h.runtime % 60}m` }) : null,
      ...(h.genres || []).slice(0, 2).map(g => el('span', { text: g }))),
    el('p', { class: 'hero-desc', text: h.overview }),
    el('div', { class: 'btns' },
      el('a', { class: 'btn btn-white', href: yt(h), target: '_blank', rel: 'noopener', text: '▶ Trailer' }),
      el('button', { class: 'btn btn-grey', text: 'ⓘ More Info', onclick: () => openMovie(h.id) }),
      addBtn)));
  return hero;
}

let personalHost = null;
async function refreshPersonalRows() {
  if (!personalHost) return;
  personalHost.replaceChildren();
  if (!myList.length) return;
  try {
    const [mine, picks] = await Promise.all([
      post('/api/cards', { ids: myList }),
      post('/api/profile', { ids: myList, n: 20 })]);
    personalHost.append(rowEl('My List', mine.movies.slice().reverse()));
    if (picks.movies.length) personalHost.append(rowEl('Top Picks for You', picks.movies, { showMatch: true, sub: `based on ${myList.length} title${myList.length > 1 ? 's' : ''} in your list` }));
    const last = myList[myList.length - 1];
    const d = await api(`/api/movie/${last}?n=20`);
    personalHost.append(rowEl(`Because you added ${d.title}`, d.similar, { showMatch: true }));
  } catch (e) { console.warn(e); }
}

async function showHome() {
  setActive('home'); window.scrollTo(0, 0);
  app.replaceChildren(el('div', { class: 'loading', style: 'padding-top:40vh' }, el('div', { class: 'spinner' }), 'Loading…'));
  try {
    const [d, live] = await Promise.all([api('/api/home'), api('/api/live').catch(() => ({ enabled: false, rows: [], hero: null }))]);
    const menu = $('#genreMenu');
    if (!menu.childElementCount) {
      menu.append(el('div', { class: 'dd-menu-inner' }, d.genres.map(g =>
        el('a', { href: '#', text: g.label, onclick: e => { e.preventDefault(); showGenre(g.name); } }))));
    }
    personalHost = el('div');
    const liveRows = live.enabled ? live.rows.map(r => rowEl(r.title, r.movies)) : [];
    const rows = el('div', { class: 'rows' }, personalHost, liveRows, d.rows.map(r => rowEl(r.title, r.movies)));
    app.replaceChildren(heroEl(live.enabled && live.hero ? live.hero : d.hero), rows);
    if (!live.enabled) showKeyBanner(live.error === 'bad_key'
      ? 'Your TMDB API key was rejected. Check the key in your .env file.'
      : 'Add a free TMDB API key (see README) to see latest & upcoming movies and fresh posters.');
    refreshPersonalRows();
  } catch (e) {
    app.replaceChildren(el('div', { class: 'view' }, el('div', { class: 'empty', text: 'Could not load movies. Is the server running?' })));
  }
}

function gridView(title, sub, movies, showMatch = false) {
  app.replaceChildren(el('div', { class: 'view' }, el('h1', { text: title }), el('div', { class: 'sub', text: sub }),
    movies.length ? el('div', { class: 'grid' }, movies.map(m => movieCard(m, showMatch)))
      : el('div', { class: 'empty', text: 'Nothing here yet.' })));
}

async function showGenre(name) {
  setActive('genre'); window.scrollTo(0, 0);
  const d = await api(`/api/genre/${encodeURIComponent(name)}`);
  gridView(d.title, 'Popular, well-rated picks', d.movies);
}

async function showMyList() {
  setActive('mylist'); window.scrollTo(0, 0);
  if (!myList.length) {
    app.replaceChildren(el('div', { class: 'view' }, el('h1', { text: 'My List' }),
      el('div', { class: 'empty', text: 'Add movies with the + button and FlimFlix Match will build recommendations from your taste.' })));
    return;
  }
  const [mine, picks] = await Promise.all([post('/api/cards', { ids: myList }), post('/api/profile', { ids: myList, n: 24 })]);
  const view = el('div', { class: 'view' }, el('h1', { text: 'My List' }),
    el('div', { class: 'sub', text: `${myList.length} title${myList.length > 1 ? 's' : ''}` }),
    el('div', { class: 'grid' }, mine.movies.slice().reverse().map(m => movieCard(m))),
    el('h1', { style: 'margin:44px 0 6px', text: 'Recommended from your list' }),
    el('div', { class: 'sub', text: 'Movies closest to the average taste of everything you added' }),
    el('div', { class: 'grid' }, picks.movies.map(m => movieCard(m, true))));
  app.replaceChildren(view);
}

let searchSeq = 0;
async function showSearch(q) {
  setActive('search');
  const seq = ++searchSeq;
  const d = await api(`/api/search?q=${encodeURIComponent(q)}`);
  if (seq !== searchSeq) return;
  gridView(d.movies.length ? `Results for “${q}”` : `No results for “${q}”`,
    d.movies.length ? 'Click a title, then scroll to “More Like This”' : 'Try another title', d.movies);
}

/* ------------------------------------------------------------------- modal */
const mb = $('#modalBackdrop'), modal = $('#modal');
const modalStack = [];
function closeModal() {
  mb.classList.add('hidden'); document.body.style.overflow = ''; modal.replaceChildren(); modalStack.length = 0;
}
mb.addEventListener('click', e => { if (e.target === mb) closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

async function openMovie(id) {
  mb.classList.remove('hidden'); document.body.style.overflow = 'hidden';
  modal.replaceChildren(el('div', { class: 'loading', style: 'padding:120px 0' }, el('div', { class: 'spinner' }), 'Loading…'));
  mb.scrollTop = 0;
  let m;
  try { m = await api(`/api/movie/${id}?n=18`); }
  catch (e) {
    modal.replaceChildren(el('button', { class: 'm-close', 'aria-label': 'Close', text: '✕', onclick: closeModal }),
      el('div', { class: 'loading', style: 'padding:90px 20px' },
        el('p', { text: String(e.message) === '404' ? 'No information is available for this movie.' : 'Could not load this movie right now (network or TMDB hiccup).' }),
        el('p', { style: 'margin-top:18px' }, el('button', { class: 'btn btn-grey', text: '↻ Try again', onclick: () => openMovie(id) }))));
    return;
  }
  try { renderModal(m); }
  catch (e) { console.error(e); modal.replaceChildren(el('div', { class: 'loading', text: 'Something went wrong showing this movie.' })); }
}

function renderModal(m) {
  const head = el('div', { class: 'm-head' });
  paintHeader(head, m, 'mp', 'blur');
  const addBtn = el('button', { class: 'btn btn-grey', 'data-add': m.id, onclick: () => toggleList(m) });
  syncAddBtn(addBtn, m.id);
  head.append(el('div', { class: 'm-head-in' }, el('h2', { text: m.title }),
    el('div', { class: 'btns' },
      el('a', { class: 'btn btn-white', href: yt(m), target: '_blank', rel: 'noopener', text: '▶ Trailer' }), addBtn)));

  const hrs = m.runtime ? `${Math.floor(m.runtime / 60)}h ${m.runtime % 60}m` : null;
  const body = el('div', { class: 'm-body' },
    el('div', {},
      el('div', { class: 'm-meta' },
        el('span', { class: 'match', text: `★ ${m.rating.toFixed(1)}` }), el('span', { text: `(${m.votes.toLocaleString()} votes)`, style: 'color:#888' }),
        isUpcoming(m) ? el('span', { class: 'badge', text: `Releases ${fmtDate(m.release_date)}` }) : (m.year ? el('span', { text: m.year }) : null), hrs ? el('span', { class: 'badge', text: hrs }) : null),
      m.tagline ? el('div', { class: 'm-tag', text: `“${m.tagline}”` }) : null,
      el('p', { class: 'm-over', text: m.overview })),
    el('div', { class: 'm-side' },
      el('div', {}, el('b', { text: 'Genres: ' }), el('span', { text: (m.genres || []).join(', ') || '—' })),
      m.collection ? el('div', {}, el('b', { text: 'Franchise: ' }), el('span', { text: m.collection })) : null,
      el('div', {}, el('b', { text: 'Language: ' }), el('span', { text: (m.original_language || '').toUpperCase() || '—' }))));

  const simCard = (s, withMatch) => {
    const pw = el('div', { class: 'pw' }, img(s.poster, s.title, s.id));
    const add = el('button', { class: 'add', 'data-add': s.id, style: 'opacity:1', onclick: e => { e.stopPropagation(); toggleList(s); } });
    syncAddBtn(add, s.id);
    pw.append(add);
    return el('div', { class: 'sim-card', onclick: () => openMovie(s.id) }, pw,
      el('div', { class: 'info' }, el('div', { class: 't', text: s.title }),
        el('div', { class: 'm' },
          // withMatch ? el('span', { class: 'match', text: `${s.match}% Match` }) : el('span', { text: s.rating ? `★ ${s.rating.toFixed(1)}` : '' }),
          withMatch && s.match != null
          ? el('span', { class: 'match', text: `${s.match}% Match` })
          : el('span', { text: s.rating ? `★ ${s.rating.toFixed(1)}` : '' }),
          el('span', { text: s.year || '' }))));
  };
  const sim = el('div', { class: 'm-sim' },
    el('h3', { text: 'More Like This' }),
    el('div', {
      class: 'why', text: m.live
        ? `New release — matched by the FlimFlix Match model against the 30,000-movie catalog by plot, themes and genres.`
        : `Recommended by the FlimFlix model — similar plot, themes, genres, studio and franchise to “${m.title}”.`
    }),
    el('div', { class: 'sim-grid' }, (m.similar || []).map(s => simCard(s, true))));
  if (m.tmdb_similar && m.tmdb_similar.length) {
    sim.append(el('h3', { style: 'margin-top:30px', text: 'Recent Titles Viewers Also Like' }),
      el('div', { class: 'why', text: 'From TMDB — newer films people pair with this one.' }),
      el('div', { class: 'sim-grid' }, m.tmdb_similar.map(s => simCard(s, false))));
  }

  modal.replaceChildren(el('button', { class: 'm-close', 'aria-label': 'Close', text: '✕', onclick: closeModal }), head, body, sim);
  mb.scrollTop = 0;
}

/* ------------------------------------------------------------------- nav */
window.addEventListener('scroll', () => $('#nav').classList.toggle('solid', window.scrollY > 40));
document.querySelectorAll('[data-nav]').forEach(a => a.addEventListener('click', e => {
  e.preventDefault(); closeModal(); closeSearch();
  a.dataset.nav === 'mylist' ? showMyList() : showHome();
}));
const sbox = $('#search'), sin = $('#searchInput');
function closeSearch() { if (!sin.value) sbox.classList.remove('open'); }
$('#searchBtn').addEventListener('click', () => { sbox.classList.add('open'); sin.focus(); });
sin.addEventListener('blur', () => setTimeout(closeSearch, 150));
let debounce;
sin.addEventListener('input', () => {
  clearTimeout(debounce);
  const q = sin.value.trim();
  debounce = setTimeout(() => { q ? showSearch(q) : showHome(); }, 300);
});

saveList();
showHome();
