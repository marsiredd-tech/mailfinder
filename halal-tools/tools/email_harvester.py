import argparse, csv, re, time, random
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser

import requests
from bs4 import BeautifulSoup

# Supabase opsiyonel
try:
    from supabase import create_client
except Exception:
    create_client = None

USER_AGENT = "HelalRestoranBot/1.0 (+https://halalfoodexplorer.netlify.app) contact:admin"
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

CANDIDATE_PATHS = [
    "/", "/contact", "/kontakt", "/impressum", "/contact-us",
    "/imprint", "/kontakt.html", "/impressum.html", "/about", "/ueber-uns", "/uber-uns"
]

def normalize_url(u: str) -> str | None:
    if not u:
        return None
    u = u.strip()
    if not u:
        return None
    if u.startswith("mailto:"):
        return None
    if u.startswith("http://") or u.startswith("https://"):
        return u
    if u.startswith("www."):
        return "https://" + u
    # çıplak domain
    if re.match(r"^[A-Za-z0-9][A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", u):
        return "https://" + u
    return None

def fetch(url: str) -> requests.Response | None:
    try:
        r = requests.get(
            url, timeout=12,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "de-TR,tr-DE,en;q=0.8"}
        )
        if r.status_code >= 400:
            return None
        return r
    except requests.RequestException:
        return None

def extract_emails_from_html(html: str) -> set[str]:
    emails = set()
    for m in EMAIL_RE.findall(html or ""):
        emails.add(m.lower())
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for a in soup.select('a[href^="mailto:"]'):
            href = a.get("href", "")
            addr = href[7:].split("?")[0].strip()
            if addr:
                emails.add(addr.lower())
    except Exception:
        pass
    # Basit temizlik
    cleaned = set()
    for e in emails:
        e = e.strip(" .;,/()[]{}<>").replace("mailto:", "")
        if re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", e):
            cleaned.add(e)
    return cleaned

def robots_allows(base: str, path: str) -> bool:
    try:
        rp = robotparser.RobotFileParser()
        rp.set_url(urljoin(base, "/robots.txt"))
        rp.read()
        return rp.can_fetch(USER_AGENT, urljoin(base, path))
    except Exception:
        # robots.txt okunamadıysa temkinli ama devam
        return True

def crawl_site(base_url: str, max_pages: int = 6) -> list[str]:
    base = normalize_url(base_url)
    if not base:
        return []
    emails: set[str] = set()

    # Önce ana sayfa
    paths = list(dict.fromkeys(CANDIDATE_PATHS))[:max_pages]

    for path in paths:
        if not robots_allows(base, path):
            continue
        url = urljoin(base, path)
        resp = fetch(url)
        if not resp:
            continue
        found = extract_emails_from_html(resp.text)
        emails |= found

        # Kısa gecikme (kibar tarama)
        time.sleep(random.uniform(1.0, 2.2))

        # Erken bırakma: bir e-posta bulduysak yeterli olabilir
        if emails:
            break

    return sorted(emails)

# ---------- CSV mod ----------
def run_csv(csv_in: str, csv_out: str, id_col="id", name_col="name", website_col="website"):
    rows_out = []
    with open(csv_in, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get(id_col) or ""
            name = row.get(name_col) or ""
            website = row.get(website_col) or ""
            emails = crawl_site(website)
            rows_out.append({
                "id": rid,
                "name": name,
                "website": website,
                "emails_found": ";".join(emails),
                "primary_email": emails[0] if emails else ""
            })
            print(f"[CSV] {name[:40]:40} | {website[:35]:35} | {rows_out[-1]['primary_email']}")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id","name","website","emails_found","primary_email"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"✓ Yazıldı: {csv_out}")

# ---------- Supabase mod ----------
def run_supabase(url: str, key: str, limit: int = 100, radius_km: float | None = None):
    if create_client is None:
        raise RuntimeError("supabase paketi yüklü değil. `pip install supabase`")

    sb = create_client(url, key)

    # İstersen burada Berlin yakınını filtreleyebilirsin; şimdilik email'i boş ve website'i dolu olanlar
    q = sb.table("restaurants").select("id,name,website,contact_email").is_("contact_email", None).not_.is_("website", None).limit(limit)
    data = q.execute().data or []
    print(f"Kontrol edilecek kayıt: {len(data)}")

    updated = 0
    for r in data:
        rid, name, website = r["id"], r.get("name",""), r.get("website","")
        emails = crawl_site(website)
        primary = emails[0] if emails else None
        print(f"[SB] {name[:40]:40} | {website[:35]:35} | {primary or '-'}")
        if primary:
            # E-postayı küçük harfe indirip yaz
            sb.table("restaurants").update({"contact_email": primary.lower()}).eq("id", rid).execute()
            updated += 1
    print(f"✓ Güncellendi (contact_email): {updated}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HelalRestoran email toplayıcı (web sitelerinden)")
    sub = ap.add_subparsers(dest="mode")

    ap_csv = sub.add_parser("csv", help="CSV gir, CSV çıkar")
    ap_csv.add_argument("--in", dest="csv_in", required=True, help="Girdi CSV (id,name,website)")
    ap_csv.add_argument("--out", dest="csv_out", required=True, help="Çıktı CSV")
    ap_csv.add_argument("--id-col", default="id")
    ap_csv.add_argument("--name-col", default="name")
    ap_csv.add_argument("--website-col", default="website")

    ap_sb = sub.add_parser("supabase", help="Supabase tablosunu güncelle")
    ap_sb.add_argument("--url", required=True, help="SUPABASE_URL")
    ap_sb.add_argument("--key", required=True, help="SUPABASE_SERVICE_ROLE veya yazma yetkili key")
    ap_sb.add_argument("--limit", type=int, default=100)

    args = ap.parse_args()

    if args.mode == "csv":
        run_csv(args.csv_in, args.csv_out, args.id_col, args.name_col, args.website_col)
    elif args.mode == "supabase":
        run_supabase(args.url, args.key, args.limit)
    else:
        ap.print_help()
