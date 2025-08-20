#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email Harvester (Supabase Entegrasyonlu)
- HTTP/2 kapalı (stabilite için)
- H2/H1 protocol fallback
- Hata dayanıklı (tek site patlasa bile süreç devam eder)
- Öncelikli sayfa taraması: contact/kontakt, impressum/imprint, privacy/datenschutz, about vb.
- Supabase "restaurants" tablosu: contact_email alanını günceller
- Parametreler: --limit, --offset, --concurrency, --per-host, --pages, --timeout, --verbose
"""

import asyncio
import argparse
import re
import sys
import json
import base64
import time
from typing import List, Set, Tuple, Optional
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
    create_client = None  # Supabase modülü yoksa tek URL modu yine çalışabilir.

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124 Safari/537.36"
)

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE
)

OBFUSCATIONS = [
    (r"\s?\(at\)\s?", "@"),
    (r"\s?\[at\]\s?", "@"),
    (r"\s?\{at\}\s?", "@"),
    (r"\s+at\s+", "@"),
    (r"\s?\(dot\)\s?", "."),
    (r"\s?\[dot\]\s?", "."),
    (r"\s?\{dot\}\s?", "."),
    (r"\s+dot\s+", "."),
    (r"\s?\(punkt\)\s?", "."),
    (r"\s+punkt\s+", "."),
]

PRIORITY_KEYWORDS = [
    "contact", "kontakt", "impressum", "imprint",
    "privacy", "datenschutz", "legal", "about",
    "colophon", "colofon", "colofon", "colofone",
    "mentions", "mentions-legales",
    "kuntakt", "iletisim", "iletişim",
]

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
    # basic sanity
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

def score_email_for_website(email: str, base_url: str) -> int:
    """
    Daha iyi seçim için basit bir skor:
    +3: eposta domaini siteyle aynıysa
    +2: local kısmı 'info|kontakt|contact|hello|mail'
    +1: 'support|team|office'
    -1: 'noreply|no-reply'
    """
    score = 0
    try:
        host = urlparse(base_url).hostname or ""
        domain = host.split(":")[0].lower()
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

class Fetcher:
    def __init__(self, timeout: float = 15.0, verbose: bool = False):
        self.verbose = verbose
        self.timeout = httpx.Timeout(timeout, connect=20.0)
        # H2 kapalı; limitler ayarlı
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
            ct = r.headers.get("content-type", "")
            if "text" not in ct and "json" not in ct and "xml" not in ct:
                return None
            return r.text
        except (H2ProtocolError, httpx.RemoteProtocolError, httpcore.ProtocolError) as e:
            # Ekstra güvenlik: ayrı bir client ile fallback (H1)
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

def extract_links_and_emails(html: str, base_url: str) -> Tuple[Set[str], Set[str]]:
    links: Set[str] = set()
    emails: Set[str] = set()
    if not html:
        return links, emails

    soup = BeautifulSoup(html, "html.parser")

    # mailto: ve sayfa metni
    for a in soup.find_all("a", href=True):
        href = a.get("href") or ""
        if href.startswith("mailto:"):
            em = href[7:].strip()
            # mailto:info@example.com?subject= => temizle
            if "?" in em:
                em = em.split("?", 1)[0].strip()
            if em:
                emails.add(em)
        else:
            absu = urljoin(base_url, href)
            links.add(absu)

    # JSON-LD içindeki eposta
    for s in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(s.text.strip())
            def extract_from_obj(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k.lower() == "email" and isinstance(v, str):
                            em = v.strip()
                            if em.startswith("mailto:"):
                                em = em[7:]
                            emails.add(em)
                        else:
                            extract_from_obj(v)
                elif isinstance(o, list):
                    for it in o:
                        extract_from_obj(it)
            extract_from_obj(data)
        except Exception:
            pass

    # Metinden çıkar
    text = soup.get_text(" ", strip=True)
    emails |= extract_emails_from_text(text)

    # Öncelikli linkleri öne alabilmek için linkleri filtrele (sadece aynı host ve öncelik adayları)
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

async def crawl_one_site(fetcher: Fetcher, base_url: str, pages: int = 6, verbose: bool = False) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Bir sitenin içinde en fazla `pages` sayfa gezerek email arar.
    Dönüş: (emails, scanned_pages, visited)
    """
    base = normalize_url(base_url)
    if not base:
        return set(), set(), set()
    visited: Set[str] = set()
    to_visit: List[str] = [base]
    found_emails: Set[str] = set()
    scanned: Set[str] = set()

    # Öncelik: kontakt/impressum vb. sayfalar
    # İlk sayfadan sonra, çıkan öncelikli linkleri kuyruğa koyar.
    while to_visit and len(scanned) < pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        html = await fetcher.get(base, url)
        if not html:
            continue
        scanned.add(url)

        links, emails = extract_links_and_emails(html, base)
        if emails:
            found_emails |= emails
            # bir email bulunca yine de 1-2 öncelikli sayfaya bakmak faydalı olabilir
            if len(scanned) >= pages:
                break

        # Yeni öncelikli linkleri ekle (BFS)
        for lk in links:
            if lk not in visited and same_host(base, lk):
                to_visit.append(lk)

    return found_emails, scanned, visited

class HostLimiter:
    """
    Aynı host için eşzamanlı istek sayısını sınırlamak (per-host)
    """
    def __init__(self, per_host: int):
        self.per_host = max(1, per_host)
        self._locks = {}

    def semaphore_for(self, url: str) -> asyncio.Semaphore:
        host = urlparse(url).netloc
        if host not in self._locks:
            self._locks[host] = asyncio.Semaphore(self.per_host)
        return self._locks[host]

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

    # Aday listesi: email yok, website var
    q = (
        sb.table("restaurants")
          .select("id,name,website,contact_email", count="exact")
          .is_("contact_email", None)
          .not_.is_("website", None)
    )
    start = offset
    end = offset + limit - 1
    resp = q.range(start, end).execute()
    candidates = resp.data or []
    total = resp.count or len(candidates)

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
        # per-host sınırı + global concurrency
        async with sem, limiter.semaphore_for(website):
            try:
                # İlerleme sinyali (isteğe bağlı yorumlayabilirsin)
                # print(f"[START] {name:40} | {website[:60]}", flush=True)
                emails, _scanned, _visited = await crawl_one_site(
                    fetcher, website, pages=pages, verbose=verbose
                )
                email = pick_best_email(emails, website)
                if email:
                    sb.table("restaurants").update({"contact_email": email}).eq("id", rid).execute()
                    print(f"[SB] {name:40} | {website:60} | {email}", flush=True)
                else:
                    print(f"[SB] {name:40} | {website:60} | -", flush=True)
                # print(f"[DONE ] {name:40} | {website[:60]}", flush=True)
            except Exception as e:
                print(f"[ERR] {name:40} | {website:60} | {e}", flush=True)

    tasks = [asyncio.create_task(worker(r)) for r in candidates]
    # Tek bir site patlasa bile süreç dursun istemiyoruz
    for t in asyncio.as_completed(tasks):
        try:
            await t
        except Exception as e:
            print(f"[TASK ERR] {e}", flush=True)

    await fetcher.close()
    print("✓ Supabase güncellemesi tamam", flush=True)

async def run_single(urls: List[str], pages: int = 6, concurrency: int = 10, per_host: int = 2, timeout: float = 15.0, verbose: bool = False):
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

    # Tekil URL modu (test/demolar için)
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
                timeout=args.timeout, verbose=bool(args.verbose), offset=args.offset
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
