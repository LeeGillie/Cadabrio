"""Prompt enhancement via web search for Cadabrio.

When enabled, extracts the real subject from the user's prompt, searches the
web for visual descriptions and reference images, then builds an accurate
image-generation prompt. If a reference image is found, the caller can use
image-guided generation (img2img) so the output actually resembles the subject.
"""

import re
import tempfile
import urllib.request
from urllib.parse import urljoin, urlparse

from loguru import logger
from PIL import Image


# Domains that rarely have useful product images — skip to save time
_SKIP_DOMAINS = (
    "youtube.com", "youtu.be", "reddit.com", "facebook.com",
    "twitter.com", "x.com", "tiktok.com", "instagram.com",
    "pinterest.com", "linkedin.com", "wikipedia.org",
)

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,image/*,*/*",
}


def search_and_enhance(prompt: str, max_results: int = 8) -> dict:
    """Search the web for the subject and build an enhanced prompt.

    Returns a dict with:
        - original: the original prompt
        - subject: the extracted subject phrase
        - search_context: raw text snippets from search results
        - enhanced: the refined prompt for image generation
        - sources: list of source titles/URLs
        - reference_image: PIL Image if a reference photo was found, else None
        - reference_source: URL of the reference image
    """
    # Step 1: Extract the actual subject from the user's instruction
    subject = _extract_subject(prompt)
    logger.info(f"Extracted subject: '{subject}' from prompt: '{prompt[:80]}'")

    # Step 2: Text search for specs/descriptions (prompt enhancement)
    search_context, sources = _text_search(subject, max_results)

    # Step 3: Separate image-focused search targeting dealer/listing pages
    # (these have the actual product photos, not spec/review pages)
    image_urls = _image_page_search(subject)
    ref_image, ref_source = _find_reference_image(image_urls)

    if not search_context and ref_image is None:
        return {
            "original": prompt,
            "subject": subject,
            "search_context": "",
            "enhanced": prompt,
            "sources": [],
            "reference_image": None,
            "reference_source": "",
        }

    # Step 4: Build a clean enhanced prompt from what we learned
    enhanced = _build_enhanced_prompt(prompt, subject, search_context)

    return {
        "original": prompt,
        "subject": subject,
        "search_context": search_context,
        "enhanced": enhanced,
        "sources": sources,
        "reference_image": ref_image,
        "reference_source": ref_source,
    }


# -------------------------------------------------------------------
# Subject extraction — turn user instructions into a search query
# -------------------------------------------------------------------

_INSTRUCTION_PREFIXES = re.compile(
    r"^(?:please\s+)?(?:create|generate|make|draw|paint|render|design|show"
    r"|produce|build)\s+"
    r"(?:an?\s+)?(?:image|picture|photo|photograph|illustration|rendering"
    r"|side\s+view(?:\s+image)?|front\s+view(?:\s+image)?|rear\s+view(?:\s+image)?"
    r"|3d\s+(?:model|render)|concept\s+art)?\s*"
    r"(?:of\s+)?(?:an?\s+)?",
    re.IGNORECASE,
)


def _extract_subject(prompt: str) -> str:
    """Strip instruction prefixes to get the actual subject for searching."""
    subject = _INSTRUCTION_PREFIXES.sub("", prompt).strip()
    if len(subject) < 10:
        subject = prompt
    return subject


# -------------------------------------------------------------------
# Web searches — two targeted queries
# -------------------------------------------------------------------

def _text_search(subject: str, max_results: int) -> tuple[str, list[str]]:
    """Search for text descriptions/specs of the subject (prompt enhancement)."""
    try:
        from ddgs import DDGS

        search_query = f"{subject} specifications exterior appearance"

        snippets = []
        sources = []
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=max_results))

        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            if body:
                snippets.append(body)
            if title and href:
                sources.append(f"{title} ({href})")

        combined = "\n".join(snippets)
        logger.info(f"Text search returned {len(snippets)} snippets for: {subject}")
        return combined, sources

    except Exception as e:
        logger.error(f"Text search failed: {e}")
        return "", []


def _image_page_search(subject: str) -> list[str]:
    """Search for pages likely to have product photos (dealer listings, etc.).

    Uses 'for sale' to bias toward dealer/listing pages which have the actual
    product photos, unlike spec/review pages.
    """
    try:
        from ddgs import DDGS

        urls = []
        # Search for dealer/listing pages — these have the product photos
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{subject} for sale", max_results=8))

        for r in results:
            href = r.get("href", "")
            if href:
                urls.append(href)

        logger.info(f"Image page search returned {len(urls)} URLs for: {subject}")
        return urls

    except Exception as e:
        logger.error(f"Image page search failed: {e}")
        return []


# -------------------------------------------------------------------
# Reference image discovery — multi-strategy page scraping
# -------------------------------------------------------------------

def _find_reference_image(
    urls: list[str],
) -> tuple[Image.Image | None, str]:
    """Try to find a reference image from the search result page URLs.

    Strategies per page (tried in order):
    1. og:image / twitter:image meta tags (high quality, representative)
    2. Large <img> tags in the page body (fallback)
    """
    if not urls:
        logger.info("No URLs to search for reference images")
        return None, ""

    pages_tried = 0
    for url in urls:
        # Skip social media and video sites — they never have useful product images
        domain = urlparse(url).netloc.lower()
        if any(skip in domain for skip in _SKIP_DOMAINS):
            continue

        pages_tried += 1
        if pages_tried > 6:
            break

        logger.info(f"Checking page {pages_tried} for reference image: {domain}")
        html = _fetch_page_html(url)
        if not html:
            continue

        # Strategy 1: Meta image tags (og:image, twitter:image)
        meta_url = _extract_meta_image(html)
        if meta_url:
            absolute = _make_absolute(meta_url, url)
            if absolute:
                logger.info(f"Found meta image tag on {domain}: {absolute[:80]}")
                img = _download_image(absolute)
                if img is not None:
                    logger.info(
                        f"Reference image from meta tag on {domain} "
                        f"({img.width}x{img.height})"
                    )
                    return img, absolute
                else:
                    logger.info(f"Meta image download failed, trying <img> tags")

        # Strategy 2: Large <img> tags from the page body
        img_urls = _extract_content_images(html, url)
        for img_url in img_urls[:5]:
            img = _download_image(img_url)
            if img is not None:
                logger.info(
                    f"Reference image from <img> tag on {domain} "
                    f"({img.width}x{img.height})"
                )
                return img, img_url

    logger.info(f"No reference image found after checking {pages_tried} pages")
    return None, ""


def _fetch_page_html(page_url: str) -> str | None:
    """Fetch the first ~100KB of a web page for parsing."""
    try:
        req = urllib.request.Request(page_url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(100_000)
            return data.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.info(f"Could not fetch page {urlparse(page_url).netloc}: {e!r}")
        return None


def _extract_meta_image(html: str) -> str | None:
    """Extract image URL from og:image, twitter:image meta tags.

    Handles both attribute orderings and multiline tags.
    """
    for prop in ("og:image", "twitter:image", "twitter:image:src"):
        escaped = re.escape(prop)
        # <meta property="og:image" content="URL">
        match = re.search(
            rf'<meta\s[^>]*(?:property|name)=["\']'
            rf'{escaped}'
            rf'["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1)
        # <meta content="URL" property="og:image"> (reversed order)
        match = re.search(
            rf'<meta\s[^>]*content=["\']([^"\']+)["\']'
            rf'[^>]*(?:property|name)=["\']'
            rf'{escaped}["\']',
            html, re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1)
    return None


def _extract_content_images(html: str, base_url: str) -> list[str]:
    """Extract likely product/content image URLs from <img> tags.

    Filters out icons, logos, spacers, and other non-content images.
    Returns absolute URLs sorted by likely relevance (larger explicit
    dimensions first).
    """
    img_pattern = re.compile(
        r'<img\s([^>]*?)src=["\']([^"\']+)["\']([^>]*?)>',
        re.IGNORECASE | re.DOTALL,
    )

    skip_words = (
        "logo", "icon", "pixel", "spacer", "avatar", "loading",
        "spinner", "badge", "flag", "arrow", "btn", "button",
        "tracking", "analytics", "ads", "banner-ad", "1x1",
    )

    candidates = []
    for match in img_pattern.finditer(html):
        attrs_before = match.group(1).lower()
        src = match.group(2)
        attrs_after = match.group(3).lower()
        all_attrs = attrs_before + attrs_after

        # Skip explicit 1px images
        if 'width="1"' in all_attrs or "width='1'" in all_attrs:
            continue
        if 'height="1"' in all_attrs or "height='1'" in all_attrs:
            continue

        src_lower = src.lower()

        # Skip non-content images
        if any(w in src_lower for w in skip_words):
            continue

        # Must look like a photo
        if not any(src_lower.endswith(ext) or ext + "?" in src_lower
                    for ext in (".jpg", ".jpeg", ".png", ".webp")):
            continue

        absolute = _make_absolute(src, base_url)
        if absolute:
            candidates.append(absolute)

    return candidates


def _make_absolute(url: str, base_url: str) -> str | None:
    """Convert a potentially relative URL to absolute."""
    if not url:
        return None
    if url.startswith("data:") or url.startswith("javascript:"):
        return None
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("http"):
        return url
    return urljoin(base_url, url)


def _download_image(url: str) -> Image.Image | None:
    """Download a single image URL, return PIL Image or None."""
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
            if len(data) < 5000:
                return None
            tmp.write(data)
        tmp.close()

        img = Image.open(tmp.name).convert("RGB")
        if img.width >= 200 and img.height >= 200:
            return img
        return None
    except Exception as e:
        logger.debug(f"Image download failed ({url[:60]}): {e!r}")
        return None


# -------------------------------------------------------------------
# Prompt building — clean, focused output
# -------------------------------------------------------------------

_SEARCH_MORE_SUFFIXES = [
    "for sale",
    "exterior photo",
    "review",
    "dealer",
    "listing",
    "photos",
]


def search_reference_candidates(
    subject: str,
    max_candidates: int = 12,
    offset: int = 0,
    known_urls: set[str] | None = None,
) -> list[dict]:
    """Search for candidate reference images using DuckDuckGo.

    Tries ``ddgs.images()`` first (returns direct image URLs, thumbnails,
    titles). If that fails with HTTP 403 (rate-limited), falls back to text
    search + og:image scraping from result pages.

    When *offset* > 0 ("Search for More"), the query is varied by appending
    a keyword suffix so DuckDuckGo returns different results.

    *known_urls* is a set of image URLs already in the gallery — duplicates
    are filtered out.

    Returns a list of candidate dicts with keys:
        - title, image (full URL), thumbnail, source, width, height
        - pil_thumbnail: PIL Image of downloaded thumbnail (or None)
    """
    if known_urls is None:
        known_urls = set()

    # Vary query for "Search for More" so we don't get duplicates
    query = subject
    if offset > 0:
        suffix_idx = (offset // max(max_candidates, 1)) % len(_SEARCH_MORE_SUFFIXES)
        query = f"{subject} {_SEARCH_MORE_SUFFIXES[suffix_idx]}"

    candidates = _search_candidates_images_api(query, max_candidates, known_urls)
    if not candidates:
        logger.info("ddgs.images() returned no results, falling back to og:image scraping")
        candidates = _search_candidates_og_fallback(query, max_candidates, known_urls)

    logger.info(f"Image search returned {len(candidates)} candidates for: {query[:60]}")
    return candidates


def _search_candidates_images_api(
    query: str, max_candidates: int, known_urls: set[str],
) -> list[dict]:
    """Primary strategy: use ddgs.images() API."""
    try:
        from ddgs import DDGS

        candidates = []
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_candidates + 4))

        for r in results:
            img_url = r.get("image", "")
            if img_url in known_urls:
                continue

            thumb_url = r.get("thumbnail", "")
            pil_thumb = download_thumbnail(thumb_url) if thumb_url else None

            candidates.append({
                "title": r.get("title", ""),
                "image": img_url,
                "thumbnail": thumb_url,
                "source": r.get("url", ""),
                "width": r.get("width", 0),
                "height": r.get("height", 0),
                "pil_thumbnail": pil_thumb,
            })
            if len(candidates) >= max_candidates:
                break

        return candidates

    except Exception as e:
        logger.warning(f"ddgs.images() failed ({e!r}), will try fallback")
        return []


def _search_candidates_og_fallback(
    query: str, max_candidates: int, known_urls: set[str],
) -> list[dict]:
    """Fallback strategy: text search → visit pages → scrape og:image tags."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} for sale", max_results=12))

        candidates = []
        pages_tried = 0
        for r in results:
            href = r.get("href", "")
            if not href:
                continue

            domain = urlparse(href).netloc.lower()
            if any(skip in domain for skip in _SKIP_DOMAINS):
                continue

            pages_tried += 1
            if pages_tried > 8:
                break

            html = _fetch_page_html(href)
            if not html:
                continue

            # Try og:image meta tag
            meta_url = _extract_meta_image(html)
            if meta_url:
                absolute = _make_absolute(meta_url, href)
                if absolute and absolute not in known_urls:
                    pil_thumb = download_thumbnail(absolute)
                    if pil_thumb is not None:
                        candidates.append({
                            "title": r.get("title", ""),
                            "image": absolute,
                            "thumbnail": absolute,
                            "source": href,
                            "width": 0,
                            "height": 0,
                            "pil_thumbnail": pil_thumb,
                        })
                        if len(candidates) >= max_candidates:
                            break
                        continue

            # Try <img> tags from the page body
            img_urls = _extract_content_images(html, href)
            for img_url in img_urls[:3]:
                if img_url in known_urls:
                    continue
                pil_thumb = download_thumbnail(img_url)
                if pil_thumb is not None:
                    candidates.append({
                        "title": r.get("title", ""),
                        "image": img_url,
                        "thumbnail": img_url,
                        "source": href,
                        "width": 0,
                        "height": 0,
                        "pil_thumbnail": pil_thumb,
                    })
                    if len(candidates) >= max_candidates:
                        break
                    break  # one image per page is enough

            if len(candidates) >= max_candidates:
                break

        return candidates

    except Exception as e:
        logger.error(f"og:image fallback search failed: {e}")
        return []


def download_thumbnail(url: str, timeout: int = 8) -> Image.Image | None:
    """Download a thumbnail image for gallery preview. Returns PIL Image or None."""
    try:
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(200_000)  # thumbnails are small
            if len(data) < 500:
                return None
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Resize to consistent thumbnail size for gallery
        img.thumbnail((200, 150), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        logger.debug(f"Thumbnail download failed ({url[:60]}): {e!r}")
        return None


def download_full_image(url: str, timeout: int = 15) -> Image.Image | None:
    """Download a full-resolution image. Returns PIL Image or None."""
    try:
        req = urllib.request.Request(url, headers=_HTTP_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            if len(data) < 5000:
                return None
        import io
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if img.width >= 200 and img.height >= 200:
            return img
        return None
    except Exception as e:
        logger.debug(f"Full image download failed ({url[:60]}): {e!r}")
        return None


def _build_enhanced_prompt(original: str, subject: str, context: str) -> str:
    """Build an enhanced image generation prompt from search context.

    Strategy: Use the original subject as the core, then add only
    confirmed visual attributes from search results. Don't inject
    raw search text — extract structured attributes only.
    """
    core = subject.rstrip("., ")

    attrs = _extract_visual_attributes(context, subject)

    parts = [core]

    if attrs.get("type"):
        parts[0] = f"{core}, {attrs['type']}"

    detail_parts = []
    if attrs.get("colors"):
        detail_parts.append(f"{', '.join(attrs['colors'])} color scheme")
    if attrs.get("materials"):
        detail_parts.append(f"{', '.join(attrs['materials'])} construction")
    if attrs.get("dimensions"):
        detail_parts.append(attrs["dimensions"])
    if attrs.get("features"):
        detail_parts.append(", ".join(attrs["features"][:3]))

    if detail_parts:
        parts.append(", ".join(detail_parts))

    parts.append("highly detailed, photorealistic, sharp focus, studio lighting")

    enhanced = ", ".join(parts)

    if len(enhanced) > 400:
        enhanced = enhanced[:400].rsplit(",", 1)[0]

    return enhanced


def _extract_visual_attributes(context: str, subject: str) -> dict:
    """Extract structured visual attributes from search snippets."""
    if not context:
        return {}

    cl = context.lower()
    subj_lower = subject.lower()

    # --- Subject type identification ---
    type_map = [
        (["travel trailer", "camper trailer"], "travel trailer"),
        (["bumper pull", "bumper-pull"], "bumper-pull travel trailer"),
        (["fifth wheel", "5th wheel"], "fifth-wheel trailer"),
        (["motorhome", "motor home", "class a", "class c"], "motorhome"),
        (["toy hauler"], "toy hauler trailer"),
        (["camper van", "van conversion", "class b"], "camper van"),
        (["pop-up camper", "popup camper", "tent trailer"], "pop-up camper"),
        (["sedan", "4-door"], "sedan"),
        (["suv", "sport utility", "crossover"], "SUV"),
        (["pickup truck", "pick-up truck"], "pickup truck"),
        (["motorcycle", "motorbike"], "motorcycle"),
        (["sailboat", "sailing"], "sailboat"),
        (["pontoon boat", "pontoon"], "pontoon boat"),
        (["fishing boat", "bass boat"], "fishing boat"),
        (["house", "residence", "single family"], "house"),
        (["cabin", "log cabin"], "cabin"),
        (["3d printer", "3-d printer"], "3D printer"),
        (["drone", "quadcopter", "uav"], "drone"),
    ]

    subject_type = ""
    best_hits = 0
    for keywords, label in type_map:
        hits = sum(1 for kw in keywords if kw in cl or kw in subj_lower)
        if hits > best_hits:
            subject_type = label
            best_hits = hits

    # --- Colors ---
    color_names = [
        "white", "red", "blue", "black", "silver", "gray", "grey",
        "tan", "beige", "green", "yellow", "orange", "brown",
        "burgundy", "maroon", "navy", "cream", "gold",
    ]
    found_colors = []
    for c in color_names:
        if c in subj_lower:
            found_colors.append(c)
    for c in color_names:
        if c in cl and c not in found_colors:
            if subject_type and subject_type.split()[0] in cl:
                found_colors.append(c)
            if len(found_colors) >= 4:
                break

    # --- Materials ---
    material_keywords = [
        "fiberglass", "aluminum", "steel", "carbon fiber", "wood",
        "composite", "laminated", "gel coat", "vinyl",
    ]
    found_materials = [m for m in material_keywords if m in cl][:2]

    # --- Dimensions ---
    dim_pattern = re.compile(
        r'(\d{1,3}(?:\.\d)?)\s*(?:feet|ft|foot)\s*(?:long|length)?',
        re.IGNORECASE,
    )
    dim_match = dim_pattern.search(context)
    dimensions = f"{dim_match.group(1)} feet long" if dim_match else ""

    # --- Notable features ---
    feature_keywords = [
        ("slide-out", "slide-out"), ("slideout", "slide-out"),
        ("awning", "awning"), ("roof rack", "roof rack"),
        ("lift kit", "lifted suspension"), ("lifted", "lifted suspension"),
        ("off-road", "off-road package"), ("all-terrain", "all-terrain"),
        ("bunk bed", "bunk beds"), ("queen bed", "queen bed"),
        ("solar panel", "solar panels"), ("a/c", "air conditioning"),
    ]
    found_features = []
    for keyword, label in feature_keywords:
        if keyword in cl or keyword in subj_lower:
            if label not in found_features:
                found_features.append(label)

    return {
        "type": subject_type,
        "colors": found_colors[:3],
        "materials": found_materials,
        "dimensions": dimensions,
        "features": found_features,
    }
