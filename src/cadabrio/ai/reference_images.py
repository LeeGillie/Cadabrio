"""Reference image data structures for multi-image curation workflow.

Manages a collection of candidate reference images found via web search
or added by the user. Each image can be accepted/rejected and annotated
with notes describing what aspect it represents (e.g. "shows correct color",
"shows the lift kit"). Annotations are combined into refining prompt context.
"""

from dataclasses import dataclass, field
from enum import Enum

from PIL import Image


class ReferenceStatus(Enum):
    """Status of a reference image in the curation workflow."""
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    USER_ADDED = "user_added"


@dataclass
class ReferenceImage:
    """A single reference image with metadata and curation state."""
    image: Image.Image  # PIL thumbnail or full image
    source_url: str = ""
    thumbnail_url: str = ""
    title: str = ""
    status: ReferenceStatus = ReferenceStatus.CANDIDATE
    annotation: str = ""
    index: int = 0
    # Full-resolution image (downloaded lazily when accepted)
    full_image: Image.Image | None = None


class ReferenceCollection:
    """Manages a set of reference images through the curation workflow."""

    def __init__(self):
        self._items: list[ReferenceImage] = []
        self._next_index = 0
        self._known_urls: set[str] = set()  # for deduplication

    @property
    def all(self) -> list[ReferenceImage]:
        return list(self._items)

    @property
    def accepted(self) -> list[ReferenceImage]:
        """Accepted and user-added references, sorted by index."""
        return sorted(
            [r for r in self._items
             if r.status in (ReferenceStatus.ACCEPTED, ReferenceStatus.USER_ADDED)],
            key=lambda r: r.index,
        )

    @property
    def candidates(self) -> list[ReferenceImage]:
        """Unreviewed candidate images."""
        return [r for r in self._items if r.status == ReferenceStatus.CANDIDATE]

    @property
    def best_reference(self) -> Image.Image | None:
        """First accepted image (full-res if available, else thumbnail)."""
        accepted = self.accepted
        if not accepted:
            return None
        ref = accepted[0]
        return ref.full_image if ref.full_image is not None else ref.image

    @property
    def known_urls(self) -> set[str]:
        """Set of image URLs already in the collection (for deduplication)."""
        return set(self._known_urls)

    def add_from_search(self, image: Image.Image, source_url: str = "",
                        thumbnail_url: str = "", title: str = "") -> ReferenceImage | None:
        """Add a candidate from web search results. Returns None if duplicate URL."""
        if source_url and source_url in self._known_urls:
            return None
        ref = ReferenceImage(
            image=image,
            source_url=source_url,
            thumbnail_url=thumbnail_url,
            title=title,
            status=ReferenceStatus.CANDIDATE,
            index=self._next_index,
        )
        self._next_index += 1
        self._items.append(ref)
        if source_url:
            self._known_urls.add(source_url)
        return ref

    def add_user_image(self, image: Image.Image, title: str = "User image") -> ReferenceImage:
        """Add a user-provided image (clipboard paste or file browse)."""
        ref = ReferenceImage(
            image=image,
            title=title,
            status=ReferenceStatus.USER_ADDED,
            index=self._next_index,
        )
        self._next_index += 1
        self._items.append(ref)
        return ref

    def accept(self, ref: ReferenceImage):
        """Mark a candidate as accepted."""
        if ref in self._items:
            ref.status = ReferenceStatus.ACCEPTED

    def reject(self, ref: ReferenceImage):
        """Mark a candidate as rejected."""
        if ref in self._items:
            ref.status = ReferenceStatus.REJECTED

    def clear(self):
        """Remove all references."""
        self._items.clear()
        self._known_urls.clear()
        self._next_index = 0

    def build_refining_context(self) -> str:
        """Combine annotations from accepted references into prompt text.

        Returns a string like:
        "Reference 1 shows correct exterior color. Reference 2 shows the lift kit."
        """
        parts = []
        for i, ref in enumerate(self.accepted, 1):
            if ref.annotation.strip():
                parts.append(f"Reference {i}: {ref.annotation.strip()}")
        return ". ".join(parts)

    def __len__(self) -> int:
        return len(self._items)
