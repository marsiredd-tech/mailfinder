# tools/email_harvester_fast.py
import argparse, asyncio, csv, re, time
from html import unescape
from urllib.parse import urljoin, urlparse, unquote
import urllib.robotparser as robotparser

import httpx
from bs4 import BeautifulSoup

# Opsiyonel: Supabase
try:
    from supabase import create_client
except Exception:
    create_client = None

# ---------- Ayarlar ----------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0 Safari/537.36; HelalRestoranBot/2.0 (+https://halalfoodexplorer.netlify.app)"
)
REQ_TIMEOUT = 10.0

SEED_PATHS = [
    "/impressum", "/kontakt", "/contact", "/colofon", "/colophon",
    "/imprint", "/kontakt.html", "/impressum.html",
    "/privacy", "/datenschutz", "/about", "/ueber-uns", "/uber-uns", "/"
]

LINK_KEYWORDS = [
    "contact","kontakt","impressum","imprint","colofon","colophon",
    "privacy","datenschutz","about","ueber","über","uber","email","e-mail","mail","kontaktformular","legal","agb","terms"
]

SKIP_EXT = (".jpg",".jpeg",".png",".gif",".webp",".svg",".css",".js",".ico",".mp4",".mp3",".pdf",".doc",".docx",".xls",".xlsx",".zip",".rar")

EMAIL_RE = re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.IGNORECASE)
FREE_MAIL_HOSTS = {"gmail.com","yahoo.com","hotmail.com","outlook.com","gmx.de","web.de","t-online.de","icloud.com","proton.me","protonmail.com","yandex.com"}
AGGREGATOR_HINTS = [
    "betrieben und verwaltet durch lieferando","lieferando","wolt","uber eats","ubereats","just eat","takeaway.com"
]

# ---------- Yardımcılar ----------
def normalize_url(u: str) -> str | None:
    if not u: return None
    u = u.strip()
    if not u or u.lower().startswith("mailto:"): return None
    if u.startswith(("http://","https://")): return u
    if u.startswith("www."): return "https://" + u
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", u):
        return "https://" + u
    return None

def same_host(a: str, b: str) -> bool:
    try:
        return urlparse(a).netloc.lower() == urlparse(b).netloc.lower()
    except Exception:
        return False

_robots_cache: dict[str, robotparser.RobotFileParser] = {}

def robots_allows(base_url: str, full_url: str) -> bool:
    try:
        host = urlparse(base_url).netloc.lower()
        rp = _robots_cache.get(host)
        if not rp:
            rp = robotparser.RobotFileParser()
            rp.set_url(urljoin(base_url, "/robots.txt"))
            rp.read()
            _robots_cache[host] = rp
        return rp.can_fetch(USER_AGENT, full_url)
    except Exception:
        return True

def decode_cfemail(hex_str: str) -> str | None:
    try:
        r = bytes.fromhex(hex_str)
        key = r[0]
        out = bytes([b ^ key for b in r[1:]])
        return out.decode("utf-8")
    except Exception:
        return None

OBF_PATTERNS = [
    (re.compile(r"\s*\[?\s*(?:at|ät)\s*\]?\s*", re.IGNORECASE), "@"),
    (re.compile(r"\s*\(?\s*(?:dot|punkt)\s*\)?\s*", re.IGNORECASE), "."),
    (re.compile(r"\s+at\s+", re.IGNORECASE), "@"),
    (re.compile(r"\s+dot\s+", re.IGNORECASE), "."),
]
def deobfuscate_text(txt: str) -> str:
    if not txt: return ""
    t = unescape(txt)
    t = unquote(t)
    for pat, repl in OBF_PATTERNS: t = pat.sub(repl, t)
    t = re.sub(r"\s*@\s*", "@", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    t = t.replace('" + "', "")
    return t

def extract_emails_from_html(text: str) -> set[str]:
    emails = set()
    for block in [text, unescape(text), deobfuscate_text(text)]:
        for m in EMAIL_RE.findall(block or ""):
            emails.add(m.lower())
    try:
        soup = BeautifulSoup(unescape(text) or "", "html.parser")
        for a in soup.select('a[href^="mailto:"]'):
            href = a.get("href","")
            addr = href[7:].split("?",1)[0].strip()
            addr = deobfuscate_text(unquote(addr))
            if addr: emails.add(addr.lower())
        for cf in soup.select("[data-cfemail]"):
            dec = decode_cfemail(cf.get("data-cfemail"))
            if dec: emails.add(dec.lower())
        metas = " ".join(m.get("content","") for m in soup.find_all("meta"))
        scripts = " ".join(s.get_text(" ", strip=True) for s in soup.find_all("script"))
        for block in [metas, scripts]:
            block = deobfuscate_text(block)
            for m in EMAIL_RE.findall(block or ""):
                emails.add(m.lower())
    except Exception:
        pass
    cleaned = set()
    for e in emails:
        e = e.strip(" .;,/()[]{}<>").replace("mailto:", "")
        if re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", e):
            cleaned.add(e)
    return cleaned

def discover_links(base_url: str, html_text: str, fallback_take: int = 2) -> list[str]:
    out, same_host_links = [], []
    soup = BeautifulSoup(html_text, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:"): continue
        if any(href.lower().endswith(ext) for ext in SKIP_EXT): continue
        url = urljoin(base_url, href)
        if not same_host(base_url, url): continue
        text = (a.get_text(" ", strip=True) or "").lower()
        h = href.lower()
        if any(k in h for k in LINK_KEYWORDS) or any(k in text for k in LINK_KEYWORDS):
            out.append(url)
        else:
            same_host_links.append(url)
    out = list(dict.fromkeys(out))
    if not out and same_host_links:
        for u in same_host_links:
            if len(out) >= fallback_take: break
            out.append(u)
    return out

def choose_primary_email(emails: list[str], website_url: str) -> str | None:
    if not emails: return None
    host = urlparse(website_url).netloc.lower()
    dom = host.split(":")[0].split(".")
    domain_base = ".".join(dom[-2:]) if len(dom) >= 2 else host
    for e in emails:
        try:
            if e.split("@",1)[1].lower().endswith(domain_base):
                return e
        except Exception: pass
    non_free = [e for e in emails if e.split("@")[-1].lower() not in FREE_MAIL_HOSTS]
    return (non_free[0] if non_free else emails[0])

# ---------- Asenkron çekirdek ----------
class Fetcher:
    def __init__(self, per_host: int = 3, total_concurrency: int = 30):
        limits = httpx.Limits(max_connections=total_concurrency, max_keepalive_connections=total_concurrency)
        self.client = httpx.AsyncClient(http2=True, timeout=REQ_TIMEOUT, limits=limits, headers={"User-Agent": USER_AGENT})
        self.host_semaphores: dict[str, asyncio.Semaphore] = {}
        self.per_host = per_host

    def _sem(self, url: str) -> asyncio.Semaphore:
        host = urlparse(url).netloc.lower()
        if host not in self.host_semaphores:
            self.host_semaphores[host] = asyncio.Semaphore(self.per_host)
        return self.host_semaphores[host]

    async def get(self, base: str, url: str) -> bytes | None:
        if not robots_allows(base, url): return None
        sem = self._sem(url)
        async with sem:
            try:
                r = await self.client.get(url, follow_redirects=True)
                if r.status_code >= 400: return None
                return r.content
            except httpx.HTTPError:
                return None

    async def close(self):
        await self.client.aclose()

async def crawl_one_site(fetcher: Fetcher, website: str, pages: int = 6, verbose: bool = False):
    base = normalize_url(website)
    if not base: return [], {}, False
    emails: dict[str, set[str]] = {}
    aggregator = False

    # 1) tohum sayfaları eşzamanlı
    first_batch = [urljoin(base, p) for p in SEED_PATHS][:pages]
    first_batch = list(dict.fromkeys(first_batch))

    async def fetch_and_extract(url):
        content = await fetcher.get(base, url)
        if not content and url.startswith("https://"):
            # http fallback
            http_url = "http://" + url[len("https://"):]
            content = await fetcher.get(base, http_url)
        return content, url

    # İlk dalga
    tasks = [asyncio.create_task(fetch_and_extract(u)) for u in first_batch]
    visited = set()
    queue = []

    while tasks and len(visited) < pages:
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            content, url = await d
            visited.add(url)
            if not content: 
                continue
            text = content.decode("utf-8", errors="ignore")
            if any(h in text.lower() for h in AGGREGATOR_HINTS): aggregator = True
            found = extract_emails_from_html(text)
            if found:
                for e in found: emails.setdefault(e, set()).add(url)
            else:
                # e-posta yoksa keşif linkleri
                if len(visited) + len(tasks) < pages:
                    for ln in discover_links(base, text, fallback_take=2):
                        if ln not in visited and ln not in queue:
                            queue.append(ln)
        # yeni görevler ekle
        while queue and len(visited) + len(tasks) < pages:
            u = queue.pop(0)
            tasks.add(asyncio.create_task(fetch_and_extract(u)))
        # erken çıkış: e-posta bulunduysa ve çok sayfa gezmek istemiyorsan:
        if emails:
            # yorum satırını kaldırırsan ilk email bulununca durur:
            # for t in tasks: t.cancel()
            # break
            pass

    email_list = sorted(emails.keys())
    email_sources = {e: sorted(list(srcs)) for e, srcs in emails.items()}
    return email_list, email_sources, aggregator

# ---------- CSV modu ----------
async def run_csv(csv_in: str, csv_out: str, id_col="id", name_col="name", website_col="website",
                  pages: int = 6, concurrency: int = 30, per_host: int = 3, verbose: bool=False):
    rows = []
    with open(csv_in, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row.get(id_col) or "", row.get(name_col) or "", row.get(website_col) or ""))

    fetcher = Fetcher(per_host=per_host, total_concurrency=concurrency)
    results = []

    async def worker(rid, name, website):
        emails, sources, agg = await crawl_one_site(fetcher, website, pages=pages, verbose=verbose)
        primary = choose_primary_email(emails, website) if website else None
        primary_src = (sources.get(primary) or [""])[0] if primary else ""
        if verbose:
            print(f"[CSV] {name[:40]:40} | {website[:50]:50} | {primary or '-'}")
        return {
            "id": rid, "name": name, "website": website,
            "primary_email": primary or "", "email_count": len(emails),
            "emails_found": ";".join(emails), "primary_source": primary_src,
            "is_aggregator": 1 if agg else 0
        }

    tasks = [asyncio.create_task(worker(*r)) for r in rows]
    for t in asyncio.as_completed(tasks):
        results.append(await t)

    await fetcher.close()

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","name","website","primary_email","email_count","emails_found","primary_source","is_aggregator"])
        writer.writeheader()
        writer.writerows(results)
    print(f"✓ Yazıldı: {csv_out}")

# ---------- Supabase modu ----------
async def run_supabase(url: str, key: str, limit: int = 100, pages: int = 6, concurrency: int = 30, per_host: int = 3, verbose: bool=False):
    if create_client is None:
        raise RuntimeError("supabase paketi yüklü değil. `pip install supabase`")
    sb = create_client(url, key)

    data = sb.table("restaurants") \
             .select("id,name,website,contact_email") \
             .is_("contact_email", None) \
             .not_.is_("website", None) \
             .limit(limit).execute().data or []

    fetcher = Fetcher(per_host=per_host, total_concurrency=concurrency)

    async def worker(r):
        rid, name, website = r["id"], r.get("name",""), r.get("website","")
        emails, _sources, _agg = await crawl_one_site(fetcher, website, pages=pages, verbose=verbose)
        primary = choose_primary_email(emails, website)
        if verbose:
            print(f"[SB] {name[:40]:40} | {website[:50]:50} | {primary or '-'}")
        if primary:
            sb.table("restaurants").update({"contact_email": primary.lower()}).eq("id", rid).execute()

    tasks = [asyncio.create_task(worker(r)) for r in data]
    for t in asyncio.as_completed(tasks):
        await t

    await fetcher.close()
    print("✓ Supabase güncellemesi tamam")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="FAST email harvester (HTTP/2, asyncio)")
    sub = ap.add_subparsers(dest="mode")

    ap_csv = sub.add_parser("csv", help="CSV gir, CSV çıkar")
    ap_csv.add_argument("--in", dest="csv_in", required=True)
    ap_csv.add_argument("--out", dest="csv_out", required=True)
    ap_csv.add_argument("--id-col", default="id")
    ap_csv.add_argument("--name-col", default="name")
    ap_csv.add_argument("--website-col", default="website")
    ap_csv.add_argument("--pages", type=int, default=6)
    ap_csv.add_argument("--concurrency", type=int, default=30, help="Genel eşzamanlı site sayısı")
    ap_csv.add_argument("--per-host", type=int, default=3, help="Aynı domain için eşzamanlı istek")
    ap_csv.add_argument("--verbose", action="store_true")

    ap_sb = sub.add_parser("supabase", help="Supabase tablosunu güncelle")
    ap_sb.add_argument("--url", required=True)
    ap_sb.add_argument("--key", required=True)
    ap_sb.add_argument("--limit", type=int, default=100)
    ap_sb.add_argument("--pages", type=int, default=6)
    ap_sb.add_argument("--concurrency", type=int, default=30)
    ap_sb.add_argument("--per-host", type=int, default=3)
    ap_sb.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.mode == "csv":
        asyncio.run(run_csv(args.csv_in, args.csv_out, args.id_col, args.name_col, args.website_col,
                            pages=args.pages, concurrency=args.concurrency, per_host=args.per_host, verbose=args.verbose))
    elif args.mode == "supabase":
        asyncio.run(run_supabase(args.url, args.key, args.limit, pages=args.pages,
                                 concurrency=args.concurrency, per_host=args.per_host, verbose=args.verbose))
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
