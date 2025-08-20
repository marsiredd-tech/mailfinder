#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email Harvester (Supabase Entegrasyonlu, Derin + Hızlı)
- Öncelikli sayfalar (kontakt, impressum, privacy, about, colophon …)
- robots.txt -> sitemap(ler) (index recursion: 1 seviye)
- Cloudflare data-cfemail çözme, JSON-LD, HTML entity & obfuscation çözme
- Ters yazılmış e-posta algılama (metni reverse edip regex)
- Inline <script> ve aynı host script dosyalarından email çıkarma (limitli)
- Supabase 'restaurants.contact_email' (+ email_source_url) güncelleme
- Per-host concurrency, global concurrency, timeout, ayrıntılı log
"""

import asyncio
import argparse
import re
import sys
import json
import html as htmlmod
from typing import List, Set, Tuple, Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

try:
    from supabase import create_client
except Exception:
    create_client = None

# ---------- Ayarlar ----------
UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]
DEFAULT_TIMEOUT = 15.0
SCRIPTS_PER_SITE = 3          # aynı hosttan indirilecek JS dosyası sayısı
SITEMAP_LIMIT_PER_HOST = 40   # sitemap'tan alınıp sıraya girecek maksimum URL

PRIORITY_KEYWORDS = [
    "contact","kontakt","contact-us","kontaktformular",
    "impressum","imprint","legal","mentions","mentions-legales",
    "privacy","datenschutz","policy","datenschutzerklaerung",
    "about","ueber-uns","über-uns","uber-uns","team",
    "info","colophon","colofon","colofone",
    "iletisim","iletişim",
]
PRIORITY_PATHS = [
    "/","/contact","/contact/","/contact-us","/contact-us/",
    "/kontakt","/kontakt/","/kontaktformular","/kontaktformular/",
    "/impressum","/impressum/","/imprint","/imprint/","/legal","/legal/",
    "/privacy","/privacy/","/datenschutz","/datenschutz/","/policy","/policy/",
    "/about","/about/","/ueber-uns","/ueber-uns/","/uber-uns","/uber-uns/","/über-uns","/über-uns/",
    "/team","/team/","/colophon","/colophon/","/colofon","/colofon/",
    "/iletisim","/iletisim/","/iletişim","/iletişim/","/.well-known/security.txt",
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE)
EMAIL_SIMPLE_RE = re.compile(r"mailto:([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.IGNORECASE)

OBFUSCATIONS = [
    (r"\s?\(at\)\s?", "@"), (r"\s?\[at\]\s?", "@"), (r"\s?\{at\}\s?", "@"), (r"\s+at\s+", "@"),
    (r"\s?\(ät\)\s?", "@"), (r"\s?\[ät\]\s?", "@"), (r"\s?\{ät\}\s?", "@"), (r"\s+ät\s+", "@"),
    (r"\s?\(dot\)\s?", "."), (r"\s?\[dot\]\s?", "."), (r"\s?\{dot\}\s?", "."), (r"\s+dot\s+", "."),
    (r"\s?\(punkt\)\s?", "."), (r"\s+punkt\s+", "."), (r"\s?\(punto\)\s?", "."), (r"\s+punto\s+", "."),
]

REVERSED_TLDS = ["moc.", "gro.", "ten.", "ed.", "ku.oc.", "moc.oohay", "moc.liamg"]  # com, org, net, de, co.uk, yahoo.com, gmail.com (ters)
CF_SELECTOR = "[data-cfemail], span.__cf_email__, a.__cf_email__"


# ---------- Yardımcılar ----------
def hdrs():
    import random
    return {
        "User-Agent": random.choice(UA_LIST),
        "Accept-Language": "de-DE,de;q=0.9,tr-TR;q=0.8,en-US;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not u: return None
    if u.startswith("mailto:"): return u
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    try:
        p = urlparse(u)
        if not p.netloc: return None
        return u
    except Exception:
        return None

def same_host(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False

def root_domain(host: str) -> str:
    if not host: return ""
    parts = host.lower().split(".")
    if len(parts) >= 2: return ".".join(parts[-2:])
    return host.lower()

def deobfuscate_text(t: str) -> str:
    s = htmlmod.unescape(t)  # &#64; &commat; gibi entity'ler
    for patt, rep in OBFUSCATIONS:
        s = re.sub(patt, rep, s, flags=re.IGNORECASE)
    return s

def extract_emails_from_text(text: str) -> Set[str]:
    emails = set()
    if not text: return emails
    clean = deobfuscate_text(text)
    for m in EMAIL_RE.findall(clean):
        emails.add(m.strip().strip(".,;:|/()<>[]{}\"'"))
    # TERS METİN TARAMA (ör: ed.mial@ofni)
    rev = clean[::-1]
    for m in EMAIL_RE.findall(rev):
        em = m[::-1]  # tekrar düz çevir
        # ters TLD ipucu içeriyorsa daha olası
        if any(t in m.lower() for t in REVERSED_TLDS):
            emails.add(em.strip().strip(".,;:|/()<>[]{}\"'"))
    return emails

def cf_decode(hexstr: str) -> Optional[str]:
    try:
        r = int(hexstr[:2], 16)
        return ''.join(chr(int(hexstr[i:i+2], 16) ^ r) for i in range(2, len(hexstr), 2))
    except Exception:
        return None

def score_email_for_website(email: str, base_url: str) -> int:
    score = 0
    try:
        host = (urlparse(base_url).hostname or "").lower()
        email_domain = email.split("@", 1)[-1].lower()
        if email_domain.endswith(root_domain(host)):
            score += 5
        if email_domain == host:
            score += 2
    except Exception:
        pass
    local = email.split("@")[0].lower()
    if local in {"info","kontakt","contact","hello","mail"}: score += 2
    if local in {"support","team","office"}: score += 1
    if "noreply" in local or "no-reply" in local: score -= 2
    return score

def pick_best_email(emails: Set[str], base_url: str) -> Optional[str]:
    if not emails: return None
    ranked = sorted(emails, key=lambda e: (-score_email_for_website(e, base_url), e))
    return ranked[0]


# ---------- HTTP ----------
class Fetcher:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT, verbose: bool = False):
        self.verbose = verbose
        self.timeout = httpx.Timeout(timeout, connect=20.0)
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers=hdrs(),
            http2=False,  # stabilite için kapalı
            limits=httpx.Limits(max_keepalive_connections=40, max_connections=100),
        )

    async def close(self):
        try: await self.client.aclose()
        except Exception: pass

    async def get(self, base: Optional[str], url: str) -> Optional[str]:
        if base and url.startswith("/"):
            url = urljoin(base.rstrip("/") + "/", url)
        try:
            r = await self.client.get(url, follow_redirects=True)
            r.raise_for_status()
            ct = (r.headers.get("content-type") or "").lower()
            if "text" in ct or "html" in ct or "xml" in ct or "json" in ct:
                return r.text
            return None
        except Exception as e:
            if self.verbose: print(f"[GET ERR] {url} -> {e}", flush=True)
            return None


# ---------- HTML / Link çıkarımı ----------
def parse_sitemap_xml(xml_text: str) -> List[str]:
    urls: List[str] = []
    if not xml_text: return urls
    try:
        soup = BeautifulSoup(xml_text, "xml")
        locs = [ (loc.text or "").strip() for loc in soup.find_all("loc") ]
        urls = [u for u in locs if u]
    except Exception:
        pass
    return urls

async def discover_sitemaps(fetcher: Fetcher, base_url: str) -> List[str]:
    host = normalize_url(base_url)
    if not host: return []
    robots_url = urljoin(host.rstrip("/") + "/", "/robots.txt")
    robots = await fetcher.get(host, robots_url)
    sitemaps: List[str] = []
    if robots:
        for line in robots.splitlines():
            line = line.strip()
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                sm = normalize_url(sm) or sm
                if sm: sitemaps.append(sm)
    if not sitemaps:
        sitemaps.extend([urljoin(host, "/sitemap.xml"), urljoin(host, "/sitemap_index.xml")])
    # uniq
    seen, out = set(), []
    for u in sitemaps:
        u = normalize_url(u)
        if u and u not in seen:
            seen.add(u); out.append(u)
    return out

async def expand_sitemap(fetcher: Fetcher, sm_url: str, host: str, limit: int) -> List[str]:
    xml = await fetcher.get(sm_url, sm_url)
    if not xml: return []
    urls = parse_sitemap_xml(xml)
    # sitemap index mi?
    if any(u.lower().endswith(".xml") for u in urls) and len(urls) <= 100:
        out = []
        for sub in urls[:10]:  # 1 seviye recursion
            out.extend(await expand_sitemap(fetcher, sub, host, limit))
            if len(out) >= limit: break
        return out[:limit]
    # urlset
    res = []
    for u in urls:
        try:
            pu = urlparse(u)
            if pu.netloc == urlparse(host).netloc:
                res.append(u)
        except Exception:
            pass
        if len(res) >= limit: break
    return res[:limit]

def priority_seeds_for(base_url: str) -> List[str]:
    p = urlparse(base_url)
    root = f"{p.scheme}://{p.netloc}"
    seeds = [urljoin(root, path) for path in PRIORITY_PATHS]
    # uniq
    seen, out = set(), []
    for u in seeds:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def extract_links_emails_from_html(html: str, base_url: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    return (priority_links, all_links_same_host, emails)
    """
    pri: Set[str] = set()
    links_all: Set[str] = set()
    emails: Set[str] = set()

    soup = BeautifulSoup(html, "lxml")

    # Cloudflare gizleme
    for el in soup.select(CF_SELECTOR):
        enc = el.get("data-cfemail")
        if enc:
            dec = cf_decode(enc)
            if dec and EMAIL_RE.search(dec):
                emails.add(EMAIL_RE.search(dec).group(0))

    # mailto:
    for a in soup.find_all("a", href=True):
        href = a.get("href","").strip()
        if href.lower().startswith("mailto:"):
            m = EMAIL_SIMPLE_RE.search(href)
            if m: emails.add(m.group(1))
        else:
            absu = urljoin(base_url, href)
            links_all.add(absu)

    # JSON-LD
    for s in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(s.string or s.text or "")
            def walk(o):
                if isinstance(o, dict):
                    for k,v in o.items():
                        if isinstance(v, str) and k.lower()=="email":
                            em = v.replace("mailto:","").strip()
                            if EMAIL_RE.search(em): emails.add(EMAIL_RE.search(em).group(0))
                        else:
                            walk(v)
                elif isinstance(o, list):
                    for it in o: walk(it)
            walk(data)
            # stringte geçenler
            for em in EMAIL_RE.findall(json.dumps(data)):
                emails.add(em)
        except Exception:
            pass

    # Inline script text de tara
    for s in soup.find_all("script"):
        txt = s.string or s.text or ""
        if not txt: continue
        txt = htmlmod.unescape(txt)
        # obfuscation düzelt
        txt = deobfuscate_text(txt)
        for em in EMAIL_RE.findall(txt):
            emails.add(em)

    # Data attr / meta ipuçları
    for t in soup.find_all(attrs={"data-email": True}):
        em = (t.get("data-email") or "").strip()
        if em and EMAIL_RE.search(em):
            emails.add(EMAIL_RE.search(em).group(0))
    for m in soup.find_all("meta"):
        for attr in ("content","name","property"):
            val = (m.get(attr) or "")
            if val and EMAIL_RE.search(val):
                emails.add(EMAIL_RE.search(val).group(0))

    # Ham metin
    text = soup.get_text(" ", strip=True)
    emails |= extract_emails_from_text(text)

    # Aynı host linkleri + öncelik puanı
    base_host = urlparse(base_url).netloc
    for u in list(links_all):
        try:
            pu = urlparse(u)
            if pu.netloc != base_host:
                links_all.discard(u)
            else:
                path = (pu.path or "").lower()
                if any(k in path for k in PRIORITY_KEYWORDS):
                    pri.add(u)
        except Exception:
            links_all.discard(u)

    return pri, links_all, emails


# ---------- Tek Site Tarama ----------
async def crawl_one_site(fetcher: Fetcher, base_url: str, pages: int = 8, verbose: bool = False) -> Tuple[Set[str], str, Set[str]]:
    base = normalize_url(base_url)
    if not base: return set(), "", set()

    visited: Set[str] = set()
    scanned: Set[str] = set()
    found_emails: Set[str] = set()
    best_source: Optional[str] = None

    # 0) sitemap → öncelikli URL doldur
    to_visit: List[str] = []
    try:
        sitemaps = await discover_sitemaps(fetcher, base)
        sm_urls = []
        for sm in sitemaps[:3]:
            sm_urls.extend(await expand_sitemap(fetcher, sm, base, SITEMAP_LIMIT_PER_HOST))
        # keyword içerenler başa
        sm_urls = sorted(set(sm_urls), key=lambda u: any(k in (urlparse(u).path or "").lower() for k in PRIORITY_KEYWORDS), reverse=True)
        to_visit.extend(sm_urls)
    except Exception as e:
        if verbose: print(f"[SITEMAP ERR] {base} -> {e}", flush=True)

    # 1) Öncelikli path’ler
    to_visit = priority_seeds_for(base) + to_visit

    # 2) En sonda ana sayfa
    if base not in to_visit: to_visit.append(base)

    # 3) BFS
    extra_scripts: List[str] = []  # aynı host script src’leri (limitli)
    while to_visit and len(scanned) < pages:
        url = to_visit.pop(0)
        if url in visited: continue
        visited.add(url)

        html = await fetcher.get(base, url)
        if not html: continue

        scanned.add(url)
        pri_links, samehost_links, emails_here = extract_links_emails_from_html(html, base)

        if emails_here:
            found_emails |= emails_here
            if not best_source:
                best_source = url  # ilk bulduğumuz sayfayı kaynak say
        # inline scriptlerde yakalayamadıysa: script src tara
        if len(extra_scripts) < SCRIPTS_PER_SITE:
            soup = BeautifulSoup(html, "lxml")
            for s in soup.find_all("script", src=True):
                su = urljoin(base, s.get("src").strip())
                if same_host(base, su):
                    extra_scripts.append(su)
                if len(extra_scripts) >= SCRIPTS_PER_SITE:
                    break

        # öncelik linkleri önce
        for lk in list(pri_links) + list(samehost_links):
            if lk not in visited and same_host(base, lk):
                to_visit.append(lk)
        if len(scanned) >= pages: break

    # 4) Script dosyalarını da ara (limitli)
    for su in extra_scripts:
        js = await fetcher.get(base, su)
        if not js: continue
        # entity ve obfuscation düzelt
        js = deobfuscate_text(js)
        for em in EMAIL_RE.findall(js):
            found_emails.add(em)
            if not best_source: best_source = su

    # domain önceliğine göre en iyi e-posta
    best = pick_best_email(found_emails, base)
    return ( {best} if best else set(), best_source or base, found_emails )


# ---------- Concurrency / per-host ----------
class HostLimiter:
    def __init__(self, per_host: int):
        self.per_host = max(1, per_host)
        self._locks = {}
    def sem(self, url: str) -> asyncio.Semaphore:
        host = urlparse(url).netloc
        if host not in self._locks:
            self._locks[host] = asyncio.Semaphore(self.per_host)
        return self._locks[host]


# ---------- Supabase Akışı ----------
async def run_supabase(url: str, key: str, limit: int, offset: int,
                       pages: int, concurrency: int, per_host: int,
                       timeout: float, verbose: bool):
    if create_client is None:
        print("supabase python paketi yok. `pip install supabase`", file=sys.stderr)
        sys.exit(1)

    sb = create_client(url, key)

    q = ( sb.table("restaurants")
            .select("id,name,website,contact_email", count="exact")
            .is_("contact_email", None)
            .not_.is_("website", None)
            .order("id") )
    start, end = max(0,int(offset)), max(0,int(offset)) + int(limit) - 1

    try:
        resp = q.range(start, end).execute()
        rows = resp.data or []
        total = resp.count or (len(rows) if rows is not None else 0)
    except Exception as e:
        msg = str(e)
        if " 416" in msg or "Requested Range Not Satisfiable" in msg:
            print("Aday sayısı (bu dilimde): 0  (offset bitti)", flush=True)
            return
        raise

    print(f"Aday sayısı (bu dilimde): {len(rows)} / toplam: {total}", flush=True)
    if not rows: return

    fetcher = Fetcher(timeout=timeout, verbose=verbose)
    limiter = HostLimiter(per_host)
    sem = asyncio.Semaphore(concurrency)

    async def worker(row):
        rid = row.get("id")
        name = (row.get("name") or "")[:40]
        website = normalize_url(row.get("website") or "")
        if not rid or not website:
            if verbose: print(f"[SKIP] {name:40} | website yok/geçersiz")
            return
        async with sem, limiter.sem(website):
            try:
                best_set, source_url, all_found = await crawl_one_site(fetcher, website, pages=pages, verbose=verbose)
                best = next(iter(best_set)) if best_set else None
                if best:
                    try:
                        sb.table("restaurants").update({"contact_email": best}).eq("id", rid).execute()
                    except Exception as e:
                        print(f"  ! contact_email update err: {e}", file=sys.stderr)
                    try:
                        sb.table("restaurants").update({"email_source_url": source_url}).eq("id", rid).execute()
                    except Exception:
                        pass
                    print(f"[SB] {name:40} | {website:60} | {best}")
                else:
                    print(f"[SB] {name:40} | {website:60} | -")
            except Exception as e:
                print(f"[ERR] {name:40} | {website:60} | {e}")

    tasks = [asyncio.create_task(worker(r)) for r in rows]
    for t in asyncio.as_completed(tasks):
        try: await t
        except Exception as e: print(f"[TASK ERR] {e}")
    await fetcher.close()
    print("✓ Supabase güncellemesi tamam", flush=True)


# ---------- Tekil URL modu ----------
async def run_single(urls: List[str], pages: int, concurrency: int, per_host: int, timeout: float, verbose: bool):
    fetcher = Fetcher(timeout=timeout, verbose=verbose)
    limiter = HostLimiter(per_host)
    sem = asyncio.Semaphore(concurrency)

    async def one(u):
        u = normalize_url(u)
        if not u:
            print(f"{u}\t-\t(geçersiz)"); return
        async with sem, limiter.sem(u):
            try:
                best_set, source, scanned_all = await crawl_one_site(fetcher, u, pages=pages, verbose=verbose)
                best = next(iter(best_set)) if best_set else "-"
                print(f"{u}\t{best}\t({len(scanned_all)} sayfa)  src={source}")
            except Exception as e:
                print(f"{u}\t-\t(hata:{e})")

    await asyncio.gather(*(one(u) for u in urls))
    await fetcher.close()


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Email harvester (derin tarama)")
    sub = ap.add_subparsers(dest="mode", required=True)

    ps = sub.add_parser("supabase")
    ps.add_argument("--url", required=True)
    ps.add_argument("--key", required=True)
    ps.add_argument("--limit", type=int, default=100)
    ps.add_argument("--offset", type=int, default=0)
    ps.add_argument("--pages", type=int, default=10)
    ps.add_argument("--concurrency", type=int, default=16)
    ps.add_argument("--per-host", type=int, default=3)
    ps.add_argument("--timeout", type=float, default=15.0)
    ps.add_argument("--verbose", action="store_true")

    pu = sub.add_parser("single")
    pu.add_argument("urls", nargs="+")
    pu.add_argument("--pages", type=int, default=10)
    pu.add_argument("--concurrency", type=int, default=8)
    pu.add_argument("--per-host", type=int, default=2)
    pu.add_argument("--timeout", type=float, default=15.0)
    pu.add_argument("--verbose", action="store_true")

    args = ap.parse_args()
    if args.mode == "supabase":
        asyncio.run(run_supabase(args.url, args.key, args.limit, args.offset,
                                 args.pages, args.concurrency, args.per_host,
                                 args.timeout, args.verbose))
    else:
        asyncio.run(run_single(args.urls, args.pages, args.concurrency,
                               args.per_host, args.timeout, args.verbose))

if __name__ == "__main__":
    main()
