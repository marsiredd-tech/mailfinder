#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email Harvester (Supabase Entegrasyonlu, Dayanıklı, Geniş Kapsamlı Tarama)
- HTTP/2 kapalı + h2/h1 fallback
- Robots.txt -> Sitemap(ler) (index recursive)
- Öncelikli sayfalar (DE/TR/EN): kontakt/impressum/imprint/legal/datenschutz/privacy/about/ueber/hakkimizda/iletisim/colophon/team/reservierung/booking/karte/menu
- Cloudflare data-cfemail çözme
- JSON-LD, inline/harici script içindeki stringlerde e-posta desenleri
- Obfuscation çözme: (at)/(ät)/[at]/{at}/AT, dot/punkt/nokta, • ve NBSP/RTL/zero-width, ters yazım
- data-* attribute tarama, mailto:
- (Opsiyon) Yalnız öncelikli sayfalarda küçük OCR fallback (pytesseract)
- Per-host concurrency, timeout, retry/backoff, HEAD+Content-Type hızlı eleme, max-bytes sınırı
- (Opsiyon) Playwright headless fallback (yalnız öncelikli sayfalarda ve sınırlı denemeler)
- E-posta seçim skoru: domain eşleşmesi, yol önceliği, blacklist/serbest domain/3rd-party rezervasyon platformları
- Supabase REST: restaurants.contact_email + restaurants.email_source_url güncellemesi
- CSV çıkışı opsiyonu
- --offset-loop ile tüm dataset’i part part gezme
- Metrikler: taranan sayfa, bulunan e-posta, güncellenen kayıt adedi
- 416/Invalid JSON/ProtocolError durumlarını sessiz ve güvenli biçimde sürdürme

Not: $SUPABASE_URL / $SUPABASE_SERVICE_ROLE env değerlerini hiçbir koşulda loglamaz.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import dataclasses
import io
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from html import unescape
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import httpx
from bs4 import BeautifulSoup

# ---- Opsiyonel bağımlılıklar (varsa kullanacağız) ----
try:
    import tldextract  # root domain çıkarımı için
except Exception:  # pragma: no cover
    tldextract = None

try:
    from lxml import etree  # sitemap xml parse
except Exception:  # pragma: no cover
    etree = None

# Playwright ve OCR opsiyonel
_playwright_available = False
try:
    from playwright.async_api import async_playwright  # type: ignore
    _playwright_available = True
except Exception:
    pass

_ocr_available = False
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _ocr_available = True
except Exception:
    pass

# ----------------- Varsayılanlar -----------------

DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
)
DEFAULT_ACCEPT_LANG = "de-DE,de;q=0.9,tr-TR;q=0.8,en;q=0.7"
DEFAULT_TIMEOUT = 20
DEFAULT_CONCURRENCY = 12
DEFAULT_PER_HOST = 3
DEFAULT_PAGES = 12
DEFAULT_MAX_BYTES = 1_750_000  # ~1.7 MB
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.6

# Öncelikli sayfa anahtar kelimeleri (DE/TR/EN)
PRIORITY_PATHS = [
    "kontakt", "impressum", "imprint", "legal", "datenschutz", "privacy",
    "contact", "about", "ueber", "ueber-uns", "ueberuns", "colophon", "team",
    "hakkimizda", "iletisim", "kvkk",
    "reservation", "reservierung", "booking",
    "menu", "speisekarte", "karte",
]

# Arama sırasında link metninde veya URL’de geçtiğinde önem derecesi artsın
PRIORITY_HINTS = PRIORITY_PATHS + [
    "email", "e-mail", "mail", "impressum", "datenschutz", "privacy",
    "kontakt", "contact", "iletisim", "hakkimizda", "legal", "about",
]

# 3rd-party rezervasyon/booking platformları (domain eşleşmezse skor düşsün)
THIRDPARTY_BOOKING_DOMAINS = {
    "opentable.com", "thefork.com", "quandoo.de", "quandoo.com", "resmio.com",
    "bookatable.com", "sevenrooms.com", "reservation.dish.co", "tablein.com",
    "gettreatful.com", "resdiary.com",
}

# Free-mail domainleri (kabul ama biraz düşük skor)
FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com",
    "gmx.de", "web.de", "yandex.com", "proton.me", "icloud.com",
}

# Blacklist (local-part)
LOCALPART_BLACKLIST = {
    "no-reply", "noreply", "donotreply", "do-not-reply", "mailer-daemon",
}

# Email regEx ve obfuscation destekleri
EMAIL_REGEX = re.compile(
    r"(?<![\w\.\-])([A-Z0-9._%+\-]+)\s*@\s*([A-Z0-9.\-]+\.[A-Z]{2,})(?![\w\-])",
    re.IGNORECASE,
)

MAILTO_REGEX = re.compile(r"mailto:([^?\"\'>\s]+)", re.IGNORECASE)

# Zero-width vb. karakterler
ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\u2060]")

# Ters yazım tespiti için basit kontrol (örn. moc.liamg@eman)
REVERSED_CHARS = set("@.")


@dataclass
class PageResult:
    url: str
    emails: Set[str]
    source_rank: int  # Öncelikli sayfalarda daha yüksek olsun
    from_dynamic: bool = False


@dataclass
class HarvestResult:
    best_email: Optional[str]
    best_source_url: Optional[str]
    all_emails: Set[str]


# ----------------- Yardımcılar -----------------

def _root_domain(netloc: str) -> str:
    if tldextract:
        ext = tldextract.extract(netloc)
        # 'example.co.uk' gibi
        return ".".join(part for part in [ext.domain, ext.suffix] if part)
    # Fallback
    parts = netloc.split(":")[0].split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else netloc


def _normalize_text(s: str) -> str:
    s = unescape(s)
    s = ZERO_WIDTH.sub("", s)
    s = s.replace("\xa0", " ").replace("&nbsp;", " ")
    # Yaygın obfuscation normalizasyonu
    subs = [
        (r"\(?(?:\s*at\s*|ät|AT)\)?", "@"),
        (r"\[?\s*at\s*\]?", "@"),
        (r"\{?\s*at\s*\}?", "@"),
        (r"\(?(?:\s*dot\s*|punkt|nokta)\)?", "."),
        (r"\[?\s*dot\s*\]?", "."),
        (r"\{?\s*dot\s*\}?", "."),
        (r"•", "."),
        (r"\s+@\s+", "@"),
        (r"\s*\[\s*@\s*\]\s*", "@"),
        (r"\s*\(\s*@\s*\)\s*", "@"),
        (r"\s*\(\s*ät\s*\)\s*", "@"),
    ]
    for pat, repl in subs:
        s = re.sub(pat, repl, s, flags=re.IGNORECASE)
    # Bazı noktalama/boşluk sadeleştirme
    s = re.sub(r"\s+", " ", s)
    return s


def _maybe_reverse_email(s: str) -> Optional[str]:
    # 'moc.liamg@eman' gibi açıkça tersten yazılmışsa ters çevir
    if any(c in s for c in REVERSED_CHARS) and " " not in s:
        rev = s[::-1]
        if EMAIL_REGEX.search(rev):
            return rev
    return None


def _decode_cfemail(hex_str: str) -> Optional[str]:
    # Cloudflare data-cfemail çözümü
    try:
        r = bytes.fromhex(hex_str)
        key = r[0]
        decoded = bytes([b ^ key for b in r[1:]]).decode("utf-8", errors="ignore")
        # Data bazen 'something@domain' olarak gelir
        if EMAIL_REGEX.search(decoded):
            return decoded
        # Bazen içinde HTML/JS olabilir -> basit çıkarım
        m = EMAIL_REGEX.search(_normalize_text(decoded))
        return m.group(0) if m else None
    except Exception:
        return None


def _score_email(email: str, website_url: str, src_url: str) -> int:
    parsed = urlparse(website_url)
    site_root = _root_domain(parsed.netloc or "")
    email_domain = email.split("@")[-1].lower()
    local = email.split("@")[0].lower()
    src_path = urlparse(src_url).path.lower()

    score = 0
    # Domain eşleşmesi
    if site_root and site_root == _root_domain(email_domain):
        score += 30
    else:
        score -= 5

    # Free-mail
    if email_domain in FREEMAIL_DOMAINS:
        score -= 6

    # Local-part blacklist
    if local in LOCALPART_BLACKLIST or any(bad in local for bad in LOCALPART_BLACKLIST):
        score -= 25

    # İpuçları
    positives = ["info", "kontakt", "contact", "hello", "hallo", "reserv", "office"]
    if any(p in local for p in positives):
        score += 10

    # Kaynak sayfa öncelikli mi?
    if any(k in src_path for k in PRIORITY_PATHS):
        score += 15

    # 3rd-party booking domaini ise düşür
    if any(third in email_domain for third in THIRDPARTY_BOOKING_DOMAINS):
        score -= 30

    # Çok uzun local-part veya garip karakterler biraz düşsün
    if len(local) > 40:
        score -= 4

    return score


def _best_email(website_url: str, found: List[PageResult]) -> HarvestResult:
    # Tüm emailleri topla ve en iyi skoru seç
    candidate_map: Dict[str, Tuple[int, str]] = {}  # email -> (score, src)
    all_emails: Set[str] = set()

    for pr in found:
        for e in pr.emails:
            all_emails.add(e)
            sc = _score_email(e, website_url, pr.url) + pr.source_rank
            if e not in candidate_map or sc > candidate_map[e][0]:
                candidate_map[e] = (sc, pr.url)

    if not candidate_map:
        return HarvestResult(None, None, all_emails)

    best = max(candidate_map.items(), key=lambda kv: kv[1][0])
    best_email = best[0]
    best_src = best[1][1]
    return HarvestResult(best_email, best_src, all_emails)


def _is_same_site(a: str, b: str) -> bool:
    try:
        da, db = urlparse(a), urlparse(b)
        return _root_domain(da.netloc) == _root_domain(db.netloc)
    except Exception:
        return False


def _sanitize_url(u: str) -> str:
    # Fragments kaldır, js/mailto dışı, http(s) filtre
    u = urldefrag(u)[0]
    if u.startswith("mailto:") or u.startswith("javascript:"):
        return ""
    p = urlparse(u)
    if p.scheme in ("http", "https") and p.netloc:
        return u
    return ""


def _is_binary_content(content_type: str) -> bool:
    if not content_type:
        return False
    ct = content_type.lower()
    return not (
        "text/html" in ct
        or "application/xhtml+xml" in ct
        or "text/plain" in ct
    )


def _guess_rank_from_url(u: str) -> int:
    # Önceliklendirme: kontakt/impressum gibi sayfalar yüksek
    path = urlparse(u).path.lower()
    rank = 0
    for k in PRIORITY_PATHS:
        if k in path:
            rank += 25
    return rank


# ----------------- HTTP & Fetch -----------------

HTTP_RETRY_EXC = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    httpx.ConnectError,
    httpx.ReadError,
)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    max_bytes: int,
    timeout: float,
    max_retries: int,
    backoff_base: float,
    prefer_head: bool = True,
) -> Tuple[int, str, str]:
    """
    İçerik getirir. (status_code, content_type, text)
    max_bytes sınırını aşmaz; HEAD ile hızlı eleme dener.
    """
    # HEAD ile içerik türünü görmeye çalış
    if prefer_head:
        try:
            r = await client.head(url, timeout=timeout, follow_redirects=True)
            ct = r.headers.get("Content-Type", "")
            if _is_binary_content(ct):
                return r.status_code, ct, ""
        except Exception:
            pass

    delay = backoff_base
    for attempt in range(max_retries):
        try:
            r = await client.get(url, timeout=timeout, follow_redirects=True)
            ct = r.headers.get("Content-Type", "")
            if _is_binary_content(ct):
                return r.status_code, ct, ""
            # İçeriği sınırla
            if r.headers.get("Content-Length"):
                try:
                    if int(r.headers["Content-Length"]) > max_bytes:
                        # Büyük sayfa; ilk max_bytes kadarını al
                        text = (await r.aread())[:max_bytes].decode(r.encoding or "utf-8", errors="ignore")
                        return r.status_code, ct, text
                except Exception:
                    pass
            text = r.text
            if len(text) > max_bytes:
                text = text[:max_bytes]
            return r.status_code, ct, text
        except HTTP_RETRY_EXC:
            await asyncio.sleep(delay + random.random() * 0.1)
            delay *= 2
        except Exception:
            # Devam et
            await asyncio.sleep(delay)
            delay *= 2

    return 0, "", ""


# ----------------- Robots & Sitemap -----------------

async def get_robots_and_sitemaps(client: httpx.AsyncClient, base_url: str, timeout: float) -> Tuple[str, List[str]]:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    sitemaps: List[str] = []
    robots_txt = ""
    try:
        r = await client.get(robots_url, timeout=timeout)
        robots_txt = r.text
        for line in robots_txt.splitlines():
            if line.strip().lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    sitemaps.append(sm)
    except Exception:
        pass
    return robots_txt, sitemaps


def parse_sitemap_xml(xml_text: str) -> List[str]:
    urls: List[str] = []
    if not xml_text.strip():
        return urls
    try:
        if etree:
            root = etree.fromstring(xml_text.encode("utf-8"))
            # sitemapindex mi?
            if root.tag.endswith("sitemapindex"):
                for loc in root.findall(".//{*}loc"):
                    if loc.text:
                        urls.append(loc.text.strip())
            else:
                for loc in root.findall(".//{*}loc"):
                    if loc.text:
                        urls.append(loc.text.strip())
        else:
            # Basit fallback: regex ile loc’lar
            urls = re.findall(r"<loc>(.*?)</loc>", xml_text, flags=re.IGNORECASE | re.DOTALL)
            urls = [u.strip() for u in urls]
    except Exception:
        pass
    return urls


async def expand_sitemaps(client: httpx.AsyncClient, sitemaps: List[str], *, timeout: float, pages_limit: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    queue = deque(sitemaps)
    # sitemap index recursive (sınırlı)
    cap = pages_limit * 10  # sitemap çok genişse abartmayalım
    while queue and len(out) < cap:
        sm = queue.popleft()
        if sm in seen:
            continue
        seen.add(sm)
        try:
            r = await client.get(sm, timeout=timeout)
            if r.status_code != 200:
                continue
            urls = parse_sitemap_xml(r.text)
            # sitemapindex ise içindeki sitemaps’leri kuyruğa ekle
            if any(u.lower().endswith(".xml") for u in urls) and len(urls) <= 2000:
                for u in urls:
                    if u.lower().endswith(".xml"):
                        queue.append(u)
            # urlset ise URL’leri ekle
            for u in urls:
                if u.lower().startswith("http"):
                    out.append(u)
                    if len(out) >= cap:
                        break
        except Exception:
            continue
    return out[:cap]


# ----------------- HTML İçerik Analizi -----------------

def extract_emails_from_html(url: str, html: str) -> Set[str]:
    emails: Set[str] = set()
    norm = _normalize_text(html)

    # mailto:
    for m in MAILTO_REGEX.findall(norm):
        m = m.strip()
        if EMAIL_REGEX.search(m):
            emails.add(EMAIL_REGEX.search(m).group(0))

    # data-cfemail
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        soup = None

    if soup:
        for tag in soup.find_all(attrs={"data-cfemail": True}):
            dec = _decode_cfemail(tag.get("data-cfemail", ""))
            if dec:
                m = EMAIL_REGEX.search(_normalize_text(dec))
                if m:
                    emails.add(m.group(0))

        # JSON-LD
        for sc in soup.find_all("script", attrs={"type": "application/ld+json"}):
            try:
                # Bazen sayfada birden çok JSON objesi olabilir
                data_text = sc.get_text(strip=True)
                # Çok bozuk JSON’larda kaba bir "email":"..." regex’i de deneyelim
                jl = []
                try:
                    obj = json.loads(data_text)
                    jl = [obj]
                except Exception:
                    # Çoklu JSON olabilir
                    if data_text.strip().startswith("["):
                        jl = json.loads(data_text)
                for obj in (jl if isinstance(jl, list) else [jl]):
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if str(k).lower() == "email" and isinstance(v, str):
                                m = EMAIL_REGEX.search(_normalize_text(v))
                                if m:
                                    emails.add(m.group(0))
            except Exception:
                # Bozuksa regex ile ara
                for m in re.findall(r'"email"\s*:\s*"([^"]+)"', sc.get_text(), flags=re.IGNORECASE):
                    mm = EMAIL_REGEX.search(_normalize_text(m))
                    if mm:
                        emails.add(mm.group(0))

        # data-* attribute’lar ve title/alt
        for tag in soup.find_all(True):
            for attr in ("data-email", "data-mail", "data-contact", "title", "alt", "content", "value"):
                val = tag.get(attr)
                if isinstance(val, str):
                    mm = EMAIL_REGEX.search(_normalize_text(val))
                    if mm:
                        emails.add(mm.group(0))

        # Inline script/text
        for sc in soup.find_all("script"):
            if sc.get("src"):
                continue
            t = sc.get_text(" ", strip=False)
            if not t:
                continue
            t = _normalize_text(t)
            for m in EMAIL_REGEX.findall(t):
                emails.add("@".join(m))
            # Reverse yazım kontrolü
            for token in re.findall(r"[A-Za-z0-9@.\-]{8,}", t):
                rev = _maybe_reverse_email(token)
                if rev:
                    emails.add(rev)

        # görünür metin
        text_chunks = soup.get_text(" ", strip=True)
        text_norm = _normalize_text(text_chunks)
        for m in EMAIL_REGEX.findall(text_norm):
            emails.add("@".join(m))
        for token in re.findall(r"[A-Za-z0-9@.\-]{8,}", text_norm):
            rev = _maybe_reverse_email(token)
            if rev and EMAIL_REGEX.search(rev):
                emails.add(rev)

        # <img> alt/src’de bariz ipuçları (OCR başka fonksiyonda)
        for img in soup.find_all("img"):
            for attr in ("alt", "title"):
                val = img.get(attr)
                if isinstance(val, str):
                    mm = EMAIL_REGEX.search(_normalize_text(val))
                    if mm:
                        emails.add(mm.group(0))

    else:
        # soup yoksa düz metinde dene
        for m in EMAIL_REGEX.findall(norm):
            emails.add("@".join(m))
        for token in re.findall(r"[A-Za-z0-9@.\-]{8,}", norm):
            rev = _maybe_reverse_email(token)
            if rev and EMAIL_REGEX.search(rev):
                emails.add(rev)

    return {e.strip(" .;,") for e in emails if "@" in e}


async def extract_with_playwright(url: str, *, timeout_sec: float = 10.0) -> Tuple[str, Set[str]]:
    """
    Playwright ile sayfayı render edip içerikten email çıkarır.
    """
    if not _playwright_available:
        return "", set()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=int(timeout_sec * 1000), wait_until="networkidle")
            except Exception:
                # en azından load olsun
                with contextlib.suppress(Exception):
                    await page.wait_for_load_state("domcontentloaded", timeout=int(timeout_sec * 1000))
            content = await page.content()
            await browser.close()
        emails = extract_emails_from_html(url, content)
        return content, emails
    except Exception:
        return "", set()


def ocr_emails_from_html_images(base_url: str, html: str, *, max_images: int = 5, per_image_timeout: float = 5.0) -> Set[str]:
    """
    Sadece öncelikli sayfalarda ufak bir OCR denemesi.
    """
    if not (_ocr_available):
        return set()
    emails: Set[str] = set()
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return set()
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        images.append(urljoin(base_url, src))
        if len(images) >= max_images:
            break
    if not images:
        return set()

    # HTTP sync client (az sayıda resim)
    headers = {"User-Agent": DEFAULT_UA, "Accept-Language": DEFAULT_ACCEPT_LANG}
    with httpx.Client(follow_redirects=True, timeout=per_image_timeout, headers=headers) as cl:
        for img_url in images:
            try:
                r = cl.get(img_url)
                ct = r.headers.get("Content-Type", "").lower()
                if not any(x in ct for x in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif")):
                    continue
                img = Image.open(io.BytesIO(r.content))
                txt = pytesseract.image_to_string(img)
                mm = EMAIL_REGEX.findall(_normalize_text(txt))
                for m in mm:
                    emails.add("@".join(m))
            except Exception:
                continue
    return emails


# ----------------- Crawl Motoru -----------------

@dataclass
class CrawlConfig:
    pages: int = DEFAULT_PAGES
    concurrency: int = DEFAULT_CONCURRENCY
    per_host: int = DEFAULT_PER_HOST
    timeout: int = DEFAULT_TIMEOUT
    max_bytes: int = DEFAULT_MAX_BYTES
    max_retries: int = DEFAULT_MAX_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    verbose: bool = False
    headless: bool = False
    ocr: bool = False


class CrawlState:
    def __init__(self, start_url: str, cfg: CrawlConfig):
        self.start_url = start_url
        self.cfg = cfg
        self.visited: Set[str] = set()
        self.host_locks: Dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(cfg.per_host))
        self.queue: deque[str] = deque()
        self.found_pages: List[PageResult] = []
        self.pages_fetched: int = 0

    def log(self, *a):
        if self.cfg.verbose:
            print(*a, file=sys.stderr)


async def crawl_site_for_email(start_url: str, cfg: CrawlConfig) -> HarvestResult:
    st = CrawlState(start_url, cfg)

    # Canonical base
    parsed = urlparse(start_url)
    if not parsed.scheme:
        start_url = "http://" + start_url
        parsed = urlparse(start_url)

    base = f"{parsed.scheme}://{parsed.netloc}"

    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept-Language": DEFAULT_ACCEPT_LANG,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    limits = httpx.Limits(max_connections=cfg.concurrency, max_keepalive_connections=cfg.concurrency)
    async with httpx.AsyncClient(http2=False, headers=headers, limits=limits) as client:
        # Robots & sitemaps
        _, sitemaps = await get_robots_and_sitemaps(client, base, cfg.timeout)
        sitemap_urls = await expand_sitemaps(client, sitemaps, timeout=cfg.timeout, pages_limit=cfg.pages)

        # Kuyruğu doldur: öncelikli sayfalar + sitemap URL’leri + ana sayfa
        st.queue.append(base)
        for p in PRIORITY_PATHS:
            st.queue.append(urljoin(base, f"/{p}"))
        for u in sitemap_urls:
            if _is_same_site(base, u):
                st.queue.append(u)

        # BFS
        async def worker():
            nonlocal client
            while st.queue and st.pages_fetched < cfg.pages:
                url = st.queue.popleft()
                url = _sanitize_url(url)
                if not url or url in st.visited:
                    continue
                st.visited.add(url)

                host = urlparse(url).netloc
                sem = st.host_locks[host]
                async with sem:
                    status, ct, html = await fetch(
                        client, url,
                        max_bytes=cfg.max_bytes,
                        timeout=cfg.timeout,
                        max_retries=cfg.max_retries,
                        backoff_base=cfg.backoff_base,
                    )
                if status == 0:
                    continue
                if status >= 400:
                    # 404/403 vb. atla
                    continue

                st.pages_fetched += 1
                rank = _guess_rank_from_url(url)
                emails = extract_emails_from_html(url, html)

                # Playwright fallback (yalnız öncelikli sayfalar ve boş kaldıysa)
                if cfg.headless and rank >= 25 and not emails:
                    content, dyn_emails = await extract_with_playwright(url, timeout_sec=min(12, cfg.timeout))
                    if dyn_emails:
                        st.found_pages.append(PageResult(url, dyn_emails, rank, from_dynamic=True))
                    # OCR gerekirse
                    if cfg.ocr and not dyn_emails and content:
                        ocr_found = ocr_emails_from_html_images(url, content)
                        if ocr_found:
                            st.found_pages.append(PageResult(url, ocr_found, rank + 2, from_dynamic=True))
                else:
                    # OCR yalnız öncelikli sayfalarda ve headless olmasa da çalışabilir
                    if cfg.ocr and rank >= 25 and not emails:
                        ocr_found = ocr_emails_from_html_images(url, html)
                        if ocr_found:
                            st.found_pages.append(PageResult(url, ocr_found, rank + 2, from_dynamic=False))

                if emails:
                    st.found_pages.append(PageResult(url, emails, rank))

                # Yeni linkler (aynı siteden)
                try:
                    soup = BeautifulSoup(html, "html.parser")
                    links = []
                    for a in soup.find_all("a", href=True):
                        href = _sanitize_url(urljoin(url, a.get("href")))
                        if not href:
                            continue
                        if _is_same_site(base, href) and href not in st.visited:
                            # link metninde ipucu varsa öne al
                            text = (a.get_text(" ", strip=True) or "").lower()
                            hint_boost = any(h in href.lower() or h in text for h in PRIORITY_HINTS)
                            if hint_boost:
                                st.queue.appendleft(href)
                            else:
                                st.queue.append(href)
                except Exception:
                    pass

        workers = [asyncio.create_task(worker()) for _ in range(max(1, cfg.concurrency // 3))]
        await asyncio.gather(*workers)

    # En iyi email seçimi
    res = _best_email(start_url, st.found_pages)
    return res


# ----------------- Supabase REST -----------------

class SupabaseClient:
    def __init__(self, url: str, key: str, timeout: int = 20, verbose: bool = False):
        self.url = url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.verbose = verbose
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "return=minimal",
        }

    def log(self, *a):
        if self.verbose:
            print(*a, file=sys.stderr)

    def _endpoint(self, path: str) -> str:
        return f"{self.url}/rest/v1{path}"

    def get_batch(self, limit: int, offset: int) -> List[Dict[str, Any]]:
        """
        contact_email IS NULL ve website NOT NULL kayıtları getirir.
        """
        params = {
            "select": "id,name,website",
            "contact_email": "is.null",
            "website": "not.is.null",
            "order": "id.asc",
            "limit": str(limit),
            "offset": str(offset),
        }
        try:
            r = httpx.get(
                self._endpoint("/restaurants"),
                params=params,
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True,
                http2=False,
            )
            if r.status_code == 416:
                # Offset kapsam dışı
                return []
            if r.status_code >= 400:
                self.log(f"[warn] Supabase GET status={r.status_code}; devam ediliyor.")
                with contextlib.suppress(Exception):
                    _ = r.json()
                return []
            try:
                data = r.json()
                if isinstance(data, list):
                    return data
            except Exception:
                self.log("[warn] Supabase JSON parse hatası; devam.")
                return []
        except Exception as e:
            self.log(f"[warn] Supabase GET hata: {e}")
        return []

    def update_restaurant(self, rid: str, email: str, src_url: str) -> bool:
        payload = {"contact_email": email, "email_source_url": src_url}
        try:
            r = httpx.patch(
                self._endpoint("/restaurants"),
                params={"id": f"eq.{rid}"},
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
                follow_redirects=True,
                http2=False,
            )
            if r.status_code >= 400:
                self.log(f"[warn] Supabase PATCH status={r.status_code} (id={rid}).")
                return False
            return True
        except Exception as e:
            self.log(f"[warn] Supabase PATCH hata (id={rid}): {e}")
            return False


# ----------------- CLI -----------------

@dataclass
class ArgsCommon:
    pages: int
    concurrency: int
    per_host: int
    timeout: int
    max_bytes: int
    max_retries: int
    backoff_base: float
    verbose: bool
    headless: bool
    ocr: bool
    csv_out: Optional[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Email Harvester (Supabase + Single URL) — HelalRestoran",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sp = p.add_subparsers(dest="mode", required=True)

    # Supabase modu
    ps = sp.add_parser("supabase", help="Supabase'ten restoranları çekip e-posta hasadı yapar.")
    ps.add_argument("--url", default=os.getenv("SUPABASE_URL"), help="Supabase URL (env SUPABASE_URL)")
    ps.add_argument("--key", default=os.getenv("SUPABASE_SERVICE_ROLE"), help="Service Role Key (env SUPABASE_SERVICE_ROLE)")
    ps.add_argument("--limit", type=int, default=100, help="Batch limit (Supabase get)")
    ps.add_argument("--offset", type=int, default=0, help="Başlangıç offset (Supabase get)")
    ps.add_argument("--offset-loop", action="store_true", help="Bittikçe offset'i artırarak tüm dataset'i gez")
    # Ortak
    for pp in (ps,):
        pp.add_argument("--pages", type=int, default=DEFAULT_PAGES)
        pp.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
        pp.add_argument("--per-host", type=int, default=DEFAULT_PER_HOST)
        pp.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
        pp.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
        pp.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
        pp.add_argument("--backoff-base", type=float, default=DEFAULT_BACKOFF_BASE)
        pp.add_argument("--verbose", action="store_true")
        pp.add_argument("--headless", type=lambda s: s.lower() in ("1", "true", "yes"), default=False, help="Playwright fallback")
        pp.add_argument("--ocr", type=lambda s: s.lower() in ("1", "true", "yes"), default=False, help="OCR fallback (öncelikli sayfalar)")
        pp.add_argument("--csv-out", default=None, help="Bulunanları CSV'ye yaz (id,name,website,email,source_url)")

    # Tekil URL modu
    p1 = sp.add_parser("single", help="Tek bir siteyi derin tarama")
    p1.add_argument("url", help="Başlangıç URL'si (örn. https://altspandau.de)")
    for pp in (p1,):
        pp.add_argument("--pages", type=int, default=DEFAULT_PAGES)
        pp.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
        pp.add_argument("--per-host", type=int, default=DEFAULT_PER_HOST)
        pp.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
        pp.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
        pp.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
        pp.add_argument("--backoff-base", type=float, default=DEFAULT_BACKOFF_BASE)
        pp.add_argument("--verbose", action="store_true")
        pp.add_argument("--headless", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
        pp.add_argument("--ocr", type=lambda s: s.lower() in ("1", "true", "yes"), default=False)
        pp.add_argument("--csv-out", default=None)

    return p.parse_args()


def as_cfg(ns: argparse.Namespace) -> CrawlConfig:
    return CrawlConfig(
        pages=ns.pages,
        concurrency=ns.concurrency,
        per_host=ns.per_host,
        timeout=ns.timeout,
        max_bytes=ns.max_bytes,
        max_retries=ns.max_retries,
        backoff_base=ns.backoff_base,
        verbose=ns.verbose,
        headless=ns.headless,
        ocr=ns.ocr,
    )


async def run_single(ns: argparse.Namespace) -> int:
    cfg = as_cfg(ns)
    res = await crawl_site_for_email(ns.url, cfg)
    if res.best_email:
        print(json.dumps({
            "url": ns.url,
            "best_email": res.best_email,
            "email_source_url": res.best_source_url,
            "all_emails": sorted(res.all_emails),
        }, ensure_ascii=False))
        if ns.csv_out:
            with open(ns.csv_out, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["", "", ns.url, res.best_email, res.best_source_url])
        return 0
    else:
        print(json.dumps({
            "url": ns.url,
            "best_email": None,
            "email_source_url": None,
            "all_emails": sorted(res.all_emails),
        }, ensure_ascii=False))
        return 2


async def run_supabase(ns: argparse.Namespace) -> int:
    # Güvenlik: URL/KEY loglama yok!
    if not ns.url or not ns.key:
        print("[error] Supabase URL veya KEY eksik (env veya arg ile verin).", file=sys.stderr)
        return 1

    cfg = as_cfg(ns)
    sb = SupabaseClient(ns.url, ns.key, timeout=cfg.timeout, verbose=cfg.verbose)

    total_scanned = 0
    total_found = 0
    total_updated = 0

    # CSV init
    csv_writer = None
    csv_fh = None
    if ns.csv_out:
        csv_fh = open(ns.csv_out, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        # başlık yazma sorumluluğu caller'a; burada ek satırları append ediyoruz

    try:
        offset = ns.offset
        while True:
            batch = sb.get_batch(limit=ns.limit, offset=offset)
            if not batch:
                if ns.offset_loop and offset == ns.offset:
                    # İlk turda boş döndüyse muhtemelen offset > max: sessiz bit
                    break
                if not ns.offset_loop:
                    break
                # offset-loop aktifse: bitir
                break

            for row in batch:
                rid = row.get("id")
                name = (row.get("name") or "").strip()
                website = (row.get("website") or "").strip()
                if not website:
                    continue

                total_scanned += 1
                try:
                    res = await crawl_site_for_email(website, cfg)
                except Exception as e:
                    if cfg.verbose:
                        print(f"[warn] crawl hata id={rid}: {e}", file=sys.stderr)
                    continue

                if res.best_email and res.best_source_url:
                    total_found += 1
                    ok = sb.update_restaurant(rid, res.best_email, res.best_source_url)
                    if ok:
                        total_updated += 1
                        if cfg.verbose:
                            print(f"[ok] updated id={rid} name={name!r}", file=sys.stderr)
                        if csv_writer:
                            csv_writer.writerow([rid, name, website, res.best_email, res.best_source_url])

            # Sonraki sayfaya
            if not ns.offset_loop:
                break
            # Eğer batch sayısı limitten küçükse veri bitti
            if len(batch) < ns.limit:
                break
            offset += ns.limit

    finally:
        if csv_fh:
            csv_fh.close()

    # Basit metrikler (env değerlerini yazmadan)
    print(json.dumps({
        "mode": "supabase",
        "scanned": total_scanned,
        "emails_found": total_found,
        "updated": total_updated,
    }, ensure_ascii=False))
    return 0


def main() -> int:
    ns = parse_args()
    # asyncio.run
    try:
        if ns.mode == "single":
            return asyncio.run(run_single(ns))
        elif ns.mode == "supabase":
            return asyncio.run(run_supabase(ns))
        else:
            print("[error] mode bilinmiyor", file=sys.stderr)
            return 1
    except KeyboardInterrupt:
        print("\n[info] iptal edildi.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
