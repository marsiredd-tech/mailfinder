#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email Harvester (Supabase Entegrasyonlu, Geniş Kapsamlı Tarama)
- HTTP/2 kapalı + H2/H1 fallback (stabil)
- Hata dayanıklı (tek site patlasa bile süreç devam eder)
- Robots.txt → Sitemap(ler) → Öncelikli sayfalar (contact/kontakt, impressum/imprint, privacy/datenschutz, about/ueber-uns, colophon/colofon vb.)
- Cloudflare e-posta gizleme (data-cfemail) çözümü
- JSON-LD / metin/anchor’dan e-posta çıkarma + obfuscation çözme
- Supabase "restaurants.contact_email" güncellemesi
- 416 (offset aralığı kalmadı) güvenli yakalama + stabil sıralama
- Parametreler: --limit, --offset, --concurrency, --per-host, --pages, --timeout, --verbose
"""

import asyncio
import argparse
import re
import sys
import json
import time
from typing import List, Set, Tuple, Optional, Iterable
from urllib.parse import urljoin, urlparse

import httpx, httpcore
from bs4 import BeautifulSoup

try:
    from h2.exceptions import ProtocolError as H2ProtocolError
except Exception:
    class H2ProtocolError(Exception):
        pass

try:
    from supabase import create_client
except Exception:
    create_client = None  # Supabase modülü yoksa "single" modu yine çalışır.

# ------------------------------
# Sabitler ve regexler
# ------------------------------

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124 Safari/537.36"
)

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE
)

# Yaygın obfuscation desenleri
OBFUSCATIONS = [
    (r"\s?\(at\)\s?", "@"),
    (r"\s?\[at\]\s?", "@"),
    (r"\s?\{at\}\s?", "@"),
    (r"\s+at\s+", "@"),
    (r"\s?\(ät\)\s?", "@"),
    (r"\s?\[ät\]\s?", "@"),
    (r"\s?\{ät\}\s?", "@"),
    (r"\s+ät\s+", "@"),
    (r"\s?\(dot\)\s?", "."),
    (r"\s?\[dot\]\s?", "."),
    (r"\s?\{dot\}\s?", "."),
    (r"\s+dot\s+", "."),
    (r"\s?\(punkt\)\s?", "."),
    (r"\s+punkt\s+", "."),
    (r"\s?\(punto\)\s?", "."),
    (r"\s+punto\s+", "."),
    (r"\s+[\u00A0\u2007\u202F]\s*", " "),  # NBSP ve ince boşluklar
]

PRIORITY_KEYWORDS = [
    "contact", "kontakt", "contact-us", "kontaktformular",
    "impressum", "imprint", "legal", "mentions", "mentions-legales",
    "privacy", "datenschutz", "policy", "datenschutzerklaerung",
    "about", "ueber-uns", "über-uns", "uber-uns", "team",
    "info", "colophon", "colofon", "colofone",
    "kuntakt", "iletisim", "iletişim",
]

PRIORITY_PATHS = [
    "/", "/contact", "/contact/", "/contact-us", "/contact-us/",
    "/kontakt", "/kontakt/", "/kontaktformular", "/kontaktformular/",
    "/impressum", "/impressum/", "/imprint", "/imprint/", "/legal", "/legal/",
    "/privacy", "/privacy/", "/datenschutz", "/datenschutz/", "/policy", "/policy/",
    "/about", "/about/", "/ueber-uns", "/ueber-uns/", "/uber-uns", "/uber-uns/",
    "/über-uns", "/über-uns/", "/team", "/team/",
    "/colophon", "/colophon/", "/colofon", "/colofon/",
    "/kuntakt", "/kuntakt/", "/iletisim", "/iletisim/", "/iletişim", "/iletişim/",
    "/.well-known/security.txt",
]

SITEMAP_LIMIT_PER_HOST = 10  # sitemap'tan çekilecek öncelikli URL limiti

# ------------------------------
# Yardımcı fonksiyonlar
# ------------------------------

def normalize_url(u: str) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if not u:
        return None
    if u.startswith("mailto:"):
        return u
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "https://" + u
    try:
        p = urlparse(u)
        if not p.netloc:
            return None
        return u
    except Exception:
        return None

def same_host(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc == urlparse(b).netloc
    except Exception:
        return False

def deobfuscate_text(t: str) -> str:
    s = t
    for patt, rep in OBFUSCATIONS:
        s = re.sub(patt, rep, s, flags=re.IGNORECASE)
    return s

def extract_emails_from_text(text: str) -> Set[str]:
    emails = set()
    if not text:
        return emails
    clean = deobfuscate_text(text)
    for m in EMAIL_RE.findall(clean):
        m = m.strip().strip(".,;:)")
        emails.add(m)
    return emails

def cf_decode(hexstr: str) -> Optional[str]:
    """
    Cloudflare email gizleme (data-cfemail) çözümü.
    """
    try:
        r = int(hexstr[:2], 16)
        email = ''.join(chr(int(hexstr[i:i+2], 16) ^ r) for i in range(2, len(hexstr), 2))
        return email
    except Exception:
        return None

def score_email_for_website(email: str, base_url: str) -> int:
    score = 0
    try:
        host = (urlparse(base_url).hostname or "").lower()
        domain = host.split(":")[0]
        email_domain = email.split("@", 1)[-1].lower()
        if email_domain.endswith(domain):
            score += 3
    except Exception:
        pass
    local = email.split("@")[0].lower()
    if local in {"info", "kontakt", "contact", "hello", "mail"}:
        score += 2
    if local in {"support", "team", "office"}:
        score += 1
    if "noreply" in local or "no-reply" in local:
        score -= 1
    return score

def pick_best_email(emails: Set[str], base_url: str) -> Optional[str]:
    if not emails:
        return None
    ranked = sorted(emails, key=lambda e: (-score_email_for_website(e, base_url), e))
    return ranked[0]

# ------------------------------
# HTTP istemcisi (H2 kapalı + fallback)
# ------------------------------

class Fetcher:
    def __init__(self, timeout: float = 15.0, verbose: bool = False):
        self.verbose = verbose
        self.timeout = httpx.Timeout(timeout, connect=20.0)
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": UA},
            http2=False,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

    async def close(self):
        try:
            await self.client.aclose()
        except Exception:
            pass

    async def get(self, base: Optional[str], url: str) -> Optional[str]:
        if base and url.startswith("/"):
            url = urljoin(base.rstrip("/") + "/", url)
        try:
            r = await self.client.get(url, follow_redirects=True)
            r.raise_for_status()
            ct = (r.headers.get("content-type") or "").lower()
            if "text" not in ct and "json" not in ct and "xml" not in ct:
                return None
            return r.text
        except (H2ProtocolError, httpx.RemoteProtocolError, httpcore.ProtocolError) as e:
            if self.verbose:
                print(f"[H2->H1 fallback] {url} -> {e}", flush=True)
            try:
                async with httpx.AsyncClient(timeout=self.timeout, headers={"User-Agent": UA}, http2=False) as c:
                    r = await c.get(url, follow_redirects=True)
                    r.raise_for_status()
                    return r.text
            except Exception as e2:
                if self.verbose:
                    print(f"[GET ERR] {url} -> {e2}", flush=True)
                return None
        except Exception as e:
            if self.verbose:
                print(f"[GET ERR] {url} -> {e}", flush=True)
            return None

# ------------------------------
# HTML/Robots/Sitemap çıkarımı
# ------------------------------

def parse_sitemap_xml(xml_text: str) -> List[str]:
    """
    Basit sitemap parser: <urlset> veya <sitemapindex> içinden <loc> linklerini döndürür.
    """
    urls: List[str] = []
    if not xml_text:
        return urls
    try:
        soup = BeautifulSoup(xml_text, "xml")
        for loc in soup.find_all("loc"):
            u = (loc.text or "").strip()
            if u:
                urls.append(u)
    except Exception:
        pass
    return urls

async def discover_sitemaps(fetcher: Fetcher, base_url: str) -> List[str]:
    host = normalize_url(base_url)
    if not host:
        return []
    # 1) robots.txt → Sitemap:
    robots_url = urljoin(host.rstrip("/") + "/", "/robots.txt")
    robots = await fetcher.get(host, robots_url)
    sitemaps: List[str] = []
    if robots:
        for line in robots.splitlines():
            line = line.strip()
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                sm = normalize_url(sm) or sm
                if sm:
                    sitemaps.append(sm)
    # 2) Klasik yollar
    if not sitemaps:
        for candidate in ["/sitemap.xml", "/sitemap_index.xml"]:
            sitemaps.append(urljoin(host, candidate))
    # sıralı ve benzersiz
    out: List[str] = []
    seen = set()
    for u in sitemaps:
        u = normalize_url(u)
        if u and u not in seen:
            out.append(u)
            seen.add(u)
    return out

def extract_links_emails_from_html(html: str, base_url: str) -> Tuple[Set[str], Set[str]]:
    links: Set[str] = set()
    emails: Set[str] = set()
    if not html:
        return links, emails

    soup = BeautifulSoup(html, "html.parser")

    # Cloudflare obfuscation
    for span in soup.select("span.__cf_email__"):
        hexstr = span.get("data-cfemail") or ""
        em = cf_decode(hexstr) if hexstr else None
        if em:
            emails.add(em)

    # anchor href + mailto:
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if href.startswith("mailto:"):
            em = href[7:].strip()
            if "?" in em:
                em = em.split("?", 1)[0].strip()
            if em:
                emails.add(em)
        else:
            absu = urljoin(base_url, href)
            links.add(absu)

    # microdata/json-ld
    for s in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(s.text.strip())
            def walk(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if isinstance(v, str) and k.lower() == "email":
                            em = v.strip()
                            if em.startswith("mailto:"):
                                em = em[7:]
                            emails.add(em)
                        else:
                            walk(v)
                elif isinstance(o, list):
                    for it in o:
                        walk(it)
            walk(data)
        except Exception:
            pass

    # data-email / meta email ipuçları
    for t in soup.find_all(attrs={"data-email": True}):
        em = (t.get("data-email") or "").strip()
        if em:
            emails.add(em)

    # ham metin
    text = soup.get_text(" ", strip=True)
    emails |= extract_emails_from_text(text)

    # öncelikli link filtresi (aynı host + anahtar kelime)
    filtered: Set[str] = set()
    base_host = urlparse(base_url).netloc
    for u in links:
        pu = urlparse(u)
        if pu.netloc != base_host:
            continue
        path_lower = (pu.path or "").lower()
        if any(k in path_lower for k in PRIORITY_KEYWORDS):
            filtered.add(u)

    return filtered or links, emails

def priority_seeds_for(base_url: str) -> List[str]:
    seeds: List[str] = []
    p = urlparse(base_url)
    root = f"{p.scheme}://{p.netloc}"
    for path in PRIORITY_PATHS:
        seeds.append(urljoin(root, path))
    # benzersiz
    seen = set()
    out = []
    for u in seeds:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

# ------------------------------
# Tarama (tek site)
# ------------------------------

async def crawl_one_site(fetcher: Fetcher, base_url: str, pages: int = 6, verbose: bool = False) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Bir sitenin içinde en fazla `pages` sayfa gezerek e-posta arar.
    Dönüş: (emails, scanned_pages, visited)
    """
    base = normalize_url(base_url)
    if not base:
        return set(), set(), set()

    visited: Set[str] = set()
    scanned: Set[str] = set()
    found_emails: Set[str] = set()

    # Başlangıç kuyruğu (öncelik: bilinen sayfalar)
    to_visit: List[str] = []
    # 1) Öncelikli path'ler
    to_visit.extend(priority_seeds_for(base))
    # 2) robots → sitemap → öncelikli URL'ler
    try:
        sitemaps = await discover_sitemaps(fetcher, base)
        prio_from_sitemaps: List[str] = []
        for sm in sitemaps:
            xml = await fetcher.get(base, sm)
            urls = parse_sitemap_xml(xml or "")
            # sadece aynı host ve anahtar kelime içerenler
            candidates = []
            host = urlparse(base).netloc
            for u in urls:
                try:
                    pu = urlparse(u)
                    if pu.netloc != host:
                        continue
                    path_lower = (pu.path or "").lower()
                    if any(k in path_lower for k in PRIORITY_KEYWORDS):
                        candidates.append(u)
                except Exception:
                    continue
            prio_from_sitemaps.extend(candidates[:SITEMAP_LIMIT_PER_HOST])
        # sitemap önceliklerini başa ekle (front of queue)
        to_visit = prio_from_sitemaps + to_visit
    except Exception as e:
        if verbose:
            print(f"[SITEMAP ERR] {base} -> {e}", flush=True)

    # 3) En sonda ana sayfa
    if base not in to_visit:
        to_visit.append(base)

    # BFS
    while to_visit and len(scanned) < pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = await fetcher.get(base, url)
        if not html:
            continue

        scanned.add(url)
        links, emails = extract_links_emails_from_html(html, base)
        if emails:
            found_emails |= emails
            # email bulunduysa yine de sınırlı devam edelim (sayfa limitiyle)
            if len(scanned) >= pages:
                break

        # yeni linkleri kuyruğa ekle
        for lk in links:
            if lk not in visited and same_host(base, lk):
                to_visit.append(lk)

    return found_emails, scanned, visited

# ------------------------------
# Concurrency / per-host sınırı
# ------------------------------

class HostLimiter:
    def __init__(self, per_host: int):
        self.per_host = max(1, per_host)
        self._locks = {}

    def semaphore_for(self, url: str) -> asyncio.Semaphore:
        host = urlparse(url).netloc
        if host not in self._locks:
            self._locks[host] = asyncio.Semaphore(self.per_host)
        return self._locks[host]

# ------------------------------
# Supabase iş akışı
# ------------------------------

async def run_supabase(
    url: str,
    key: str,
    limit: int = 100,
    pages: int = 6,
    concurrency: int = 20,
    per_host: int = 2,
    timeout: float = 15.0,
    verbose: bool = False,
    offset: int = 0,
):
    if create_client is None:
        print("supabase python paketi yüklü değil. `pip install supabase`", file=sys.stderr)
        sys.exit(1)

    sb = create_client(url, key)

    # Aday listesi: contact_email IS NULL AND website NOT NULL
    q = (
        sb.table("restaurants")
          .select("id,name,website,contact_email", count="exact")
          .is_("contact_email", None)
          .not_.is_("website", None)
          .order("id")  # stabil sıralama (kritik)
    )

    start = max(0, int(offset))
    end   = start + int(limit) - 1

    try:
        resp = q.range(start, end).execute()
        candidates = resp.data or []
        total = resp.count or (len(candidates) if candidates is not None else 0)
    except Exception as e:
        # 416: İstenen aralık kalmadı (dataset küçüldü) → kibarca çık
        msg = str(e)
        if "code': 416" in msg or " 416" in msg or "Requested Range Not Satisfiable" in msg:
            print(f"Aday sayısı (bu dilimde): 0  (offset aralığı geçersiz)", flush=True)
            return
        raise

    print(f"Aday sayısı (bu dilimde): {len(candidates)} / toplam: {total}", flush=True)
    if not candidates:
        return

    fetcher = Fetcher(timeout=timeout, verbose=verbose)
    limiter = HostLimiter(per_host=per_host)
    sem = asyncio.Semaphore(concurrency)

    async def worker(row):
        name = (row.get("name") or "")[:40]
        website = normalize_url(row.get("website") or "")
        rid = row.get("id")
        if not website or not rid:
            if verbose:
                print(f"[SKIP] {name:40} | website yok/geçersiz", flush=True)
            return
        async with sem, limiter.semaphore_for(website):
            try:
                emails, _scanned, _visited = await crawl_one_site(
                    fetcher, website, pages=pages, verbose=verbose
                )
                email = pick_best_email(emails, website)
                if email:
                    sb.table("restaurants").update({"contact_email": email}).eq("id", rid).execute()
                    print(f"[SB] {name:40} | {website:60} | {email}", flush=True)
                else:
                    print(f"[SB] {name:40} | {website:60} | -", flush=True)
            except Exception as e:
                print(f"[ERR] {name:40} | {website:60} | {e}", flush=True)

    tasks = [asyncio.create_task(worker(r)) for r in candidates]
    for t in asyncio.as_completed(tasks):
        try:
            await t
        except Exception as e:
            print(f"[TASK ERR] {e}", flush=True)

    await fetcher.close()
    print("✓ Supabase güncellemesi tamam", flush=True)

# ------------------------------
# Tekil URL modu (test)
# ------------------------------

async def run_single(
    urls: List[str],
    pages: int = 6,
    concurrency: int = 10,
    per_host: int = 2,
    timeout: float = 15.0,
    verbose: bool = False,
):
    fetcher = Fetcher(timeout=timeout, verbose=verbose)
    limiter = HostLimiter(per_host=per_host)
    sem = asyncio.Semaphore(concurrency)

    async def one(u):
        u = normalize_url(u)
        if not u:
            print(f"{u}\t-\t(geçersiz url)")
            return
        async with sem, limiter.semaphore_for(u):
            try:
                emails, scanned, _ = await crawl_one_site(fetcher, u, pages=pages, verbose=verbose)
                email = pick_best_email(emails, u)
                print(f"{u}\t{email or '-'}\t({len(scanned)} sayfa)")
            except Exception as e:
                print(f"{u}\t-\t(hata:{e})")

    await asyncio.gather(*(one(u) for u in urls))
    await fetcher.close()

# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Email harvester")
    sub = ap.add_subparsers(dest="mode", required=True)

    # Supabase modu
    ap_sb = sub.add_parser("supabase", help="Supabase 'restaurants' tablosunu güncelle")
    ap_sb.add_argument("--url", required=True, help="Supabase URL (https://<ref>.supabase.co)")
    ap_sb.add_argument("--key", required=True, help="SERVICE_ROLE key")
    ap_sb.add_argument("--limit", type=int, default=100)
    ap_sb.add_argument("--offset", type=int, default=0)
    ap_sb.add_argument("--pages", type=int, default=6)
    ap_sb.add_argument("--concurrency", type=int, default=20)
    ap_sb.add_argument("--per-host", type=int, default=2)
    ap_sb.add_argument("--timeout", type=float, default=15.0)
    ap_sb.add_argument("--verbose", action="store_true")

    # Tekil URL modu
    ap_one = sub.add_parser("single", help="Tekil URL(ler) için e-posta ara")
    ap_one.add_argument("urls", nargs="+")
    ap_one.add_argument("--pages", type=int, default=6)
    ap_one.add_argument("--concurrency", type=int, default=10)
    ap_one.add_argument("--per-host", type=int, default=2)
    ap_one.add_argument("--timeout", type=float, default=15.0)
    ap_one.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.mode == "supabase":
        asyncio.run(
            run_supabase(
                args.url, args.key, args.limit, pages=args.pages,
                concurrency=args.concurrency, per_host=args.per_host,
                timeout=args.timeout, verbose=bool(args.verbose),
                offset=args.offset
            )
        )
    elif args.mode == "single":
        asyncio.run(
            run_single(
                args.urls, pages=args.pages, concurrency=args.concurrency,
                per_host=args.per_host, timeout=args.timeout, verbose=bool(args.verbose)
            )
        )

if __name__ == "__main__":
    main()
