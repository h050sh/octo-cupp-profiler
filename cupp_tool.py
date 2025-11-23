#!/usr/bin/env python3
"""CUPP-style wordlist generator with ethical defaults."""

import argparse
import itertools
import json
import os
import re
import sys
import time
import unicodedata
from hashlib import blake2b
from shutil import get_terminal_size
from typing import Dict, Iterable, List, Optional, Set, Tuple


DEFAULT_MIN_LENGTH = 8
DEFAULT_MAX_LENGTH = 24
FILE_LINE_LIMIT = 100_000
PREVIEW_DEFAULT = 0
SEPARATORS = ["", "-", "_", "."]
NUMBER_PATTERNS = ["1", "01", "001", "123", "321"]
SUFFIXES = ["!", "!!", "1", "12", "123", "1234"]
MIXED_SUFFIXES = ["1!", "12!", "123!", "!1", "!12", "!123", "01!", "001!", "321!"]
SPECIAL_CHARS = ["!", "@", "#", "-", "_", "."]
TRANSFORM_CHOICES = ["basic", "plus", "insane"]
BACK_TOKEN = "&&&"


class Backtrack(Exception):
    """Control flow for stepping back during prompts."""


class Policy:
    def __init__(
        self,
        min_length: int = DEFAULT_MIN_LENGTH,
        max_length: int = DEFAULT_MAX_LENGTH,
        require_upper: bool = True,
        require_lower: bool = True,
        require_digit: bool = True,
        require_special: bool = True,
        disallow_whitespace: bool = True,
        disallow_repeats: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.require_upper = require_upper
        self.require_lower = require_lower
        self.require_digit = require_digit
        self.require_special = require_special
        self.disallow_whitespace = disallow_whitespace
        self.disallow_repeats = disallow_repeats


class BloomDeduper:
    """Lightweight Bloom filter for deduplication."""

    def __init__(self, size_bits: int = 1 << 23, hash_count: int = 3) -> None:
        self.size_bits = size_bits
        self.hash_count = hash_count
        self.bits = bytearray((size_bits + 7) // 8)
        self.mask = size_bits - 1

    def _hashes(self, value: str) -> List[int]:
        digest = blake2b(value.encode("utf-8"), digest_size=16).digest()
        ints = [int.from_bytes(digest[i : i + 4], "big") for i in range(0, 12, 4)]
        return [(ints[i % len(ints)] + i * 0x9E3779B9) & self.mask for i in range(self.hash_count)]

    def check_or_add(self, value: str) -> bool:
        hits = self._hashes(value)
        seen = all(self.bits[h >> 3] & (1 << (h & 7)) for h in hits)
        if not seen:
            for h in hits:
                self.bits[h >> 3] |= 1 << (h & 7)
        return seen


class ExactDeduper:
    """Exact deduplication using hashed values; uses more memory but never drops uniques."""

    def __init__(self) -> None:
        self.seen: Set[int] = set()

    def check_or_add(self, value: str) -> bool:
        digest = blake2b(value.encode("utf-8"), digest_size=8).digest()
        marker = int.from_bytes(digest, "big")
        if marker in self.seen:
            return True
        self.seen.add(marker)
        return False


def supports_ansi() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def center_line(line: str, width: int) -> str:
    pad = max((width - len(line)) // 2, 0)
    return " " * pad + line


def build_banner_lines() -> List[str]:
    return [
        "  ____ _   _ ____  ____     _____           _        ____           _      ",
        " / ___| | | |  _ \\|  _ \\   |_   _|__   ___ | |___   / ___|___ _ __ | | ___ ",
        "| |   | | | | |_) | |_) |____| |/ _ \\ / _ \\| / __| | |   / _ \\ '_ \\| |/ _ \\",
        "| |___| |_| |  __/|  _ <_____| | (_) | (_) | \\__ \\ | |__|  __/ |_) | |  __/",
        " \\____|\\___/|_|   |_| \\_\\    |_|\\___/ \\___/|_|___/  \\____\\___| .__/|_|\\___|",
        "                                                           |_|             ",
    ]


def print_banner() -> None:
    lines = build_banner_lines()
    width = get_terminal_size(fallback=(80, 20)).columns
    color = supports_ansi()
    if color:
        start = 27
        end = 201
        span = max(len(lines) - 1, 1)
        for idx, line in enumerate(lines):
            tone = int(start + (end - start) * (idx / span))
            padded = center_line(line, width)
            print(f"\033[38;5;{tone}m{padded}\033[0m")
    else:
        for line in lines:
            print(center_line(line, width))
    print("Educational use only. Test only with explicit permission.\n")


def prompt_int(question: str, default: int) -> int:
    raw = input(f"{question} [{default}]: ").strip()
    if raw == BACK_TOKEN:
        raise Backtrack()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def prompt_yes_no(question: str, default: bool = True) -> bool:
    raw = input(f"{question} [Y/n]: ").strip().lower()
    if raw == BACK_TOKEN:
        raise Backtrack()
    if raw == "n":
        return False
    if raw == "y":
        return True
    return default


def policy_wizard(args: argparse.Namespace) -> Policy:
    min_length = args.min_length if args.min_length else DEFAULT_MIN_LENGTH
    max_length = args.max_length if args.max_length else DEFAULT_MAX_LENGTH
    print("\nType &&& then ENTER to go back to the previous policy question.")
    answers: Dict[str, object] = {}
    prompts = [
        ("min_length", lambda: prompt_int("Minimum password length? (example: 10)", answers.get("min_length", min_length))),
        ("max_length", lambda: prompt_int("Maximum password length? (example: 24)", answers.get("max_length", max_length))),
        ("require_upper", lambda: prompt_yes_no("Require at least one uppercase letter? Example: Y", True if answers.get("require_upper") is None else bool(answers["require_upper"]))),
        ("require_lower", lambda: prompt_yes_no("Require at least one lowercase letter? Example: Y", True if answers.get("require_lower") is None else bool(answers["require_lower"]))),
        ("require_digit", lambda: prompt_yes_no("Require at least one digit? Example: Y", True if answers.get("require_digit") is None else bool(answers["require_digit"]))),
        ("require_special", lambda: prompt_yes_no("Require at least one special character (!@#-_.)? Example: Y", True if answers.get("require_special") is None else bool(answers["require_special"]))),
        ("disallow_whitespace", lambda: prompt_yes_no("Disallow whitespace characters? Example: Y", True if answers.get("disallow_whitespace") is None else bool(answers["disallow_whitespace"]))),
        ("disallow_repeats", lambda: prompt_yes_no("Disallow repeating the same character consecutively? Example: Y", True if answers.get("disallow_repeats") is None else bool(answers["disallow_repeats"]))),
    ]

    idx = 0
    while idx < len(prompts):
        key, ask = prompts[idx]
        try:
            answers[key] = ask()
            idx += 1
        except Backtrack:
            if idx > 0:
                idx -= 1
                print("Going back to previous question.")
            else:
                print("Already at the first question.")

    return Policy(
        min_length=int(answers.get("min_length", min_length)),
        max_length=int(answers.get("max_length", max_length)),
        require_upper=bool(answers.get("require_upper", True)),
        require_lower=bool(answers.get("require_lower", True)),
        require_digit=bool(answers.get("require_digit", True)),
        require_special=bool(answers.get("require_special", True)),
        disallow_whitespace=bool(answers.get("disallow_whitespace", True)),
        disallow_repeats=bool(answers.get("disallow_repeats", True)),
    )


PROFILE_PROMPTS: List[Tuple[str, str]] = [
    ("personal_full_names", "Personal: full names (e.g., <subject>, sample subject)"),
    ("personal_nicknames", "Personal: nicknames (e.g., champ, ace)"),
    ("personal_birth_dates", "Personal: date(s) of birth (e.g., 01-01-2000)"),
    ("personal_birth_years", "Personal: birth year(s) (e.g., 1990, 2005)"),
    ("personal_birthplace", "Personal: birthplace (e.g., sampletown)"),
    ("personal_phone", "Personal: phone number (e.g., 1234567890)"),
    ("personal_house_number", "Personal: house number (e.g., 42)"),
    ("personal_postal_code", "Personal: postal code (e.g., 12345)"),
    ("personal_street", "Personal: street name (e.g., mainstreet)"),
    ("personal_favorite_number", "Personal: favorite number (e.g., 7)"),
    ("personal_initials", "Personal: initials (e.g., ss, ab)"),
    ("family_parents", "Family: parent names (e.g., guardian one, guardian two)"),
    ("family_partner", "Family: partner names (e.g., partner one)"),
    ("family_ex_partner", "Family: ex-partner names (e.g., example ex)"),
    ("family_children", "Family: children names including birthdays (e.g., child1 12-05-2010)"),
    ("pets", "Pets: names and optional birthdays (e.g., puppy 03/03/2018)"),
    ("favorites_teams", "Favorites: sports teams (e.g., sample united)"),
    ("favorites_jersey_numbers", "Favorites: jersey numbers (e.g., 9, 23)"),
    ("favorites_car", "Favorites: car make/model (e.g., samplecar x1)"),
    ("favorites_music", "Favorites: music artist/band (e.g., sample band)"),
    ("favorites_games", "Favorites: games (e.g., sample game)"),
    ("favorites_movies", "Favorites: movies/series (e.g., sample series)"),
    ("favorites_colors", "Favorites: colors (e.g., blue, red)"),
    ("favorites_foods", "Favorites: foods (e.g., pasta, salad)"),
    ("work_school_name", "Work/School: school name (e.g., sample academy)"),
    ("work_school_program", "Work/School: study program (e.g., computer science)"),
    ("work_school_workplace", "Work/School: workplace (e.g., sample corp)"),
    ("work_school_manager", "Work/School: manager/boss name (e.g., manager one)"),
    ("work_school_founding", "Work/School: company founding year (e.g., 1999)"),
    ("digital_emails", "Digital identifiers: email handles (e.g., example_user)"),
    ("digital_usernames", "Digital identifiers: usernames (e.g., cool_name)"),
    ("digital_gamertags", "Digital identifiers: gamertags (e.g., frag_master)"),
    ("digital_slang", "Digital identifiers: frequently used slang words or phrases (e.g., gg, glhf)"),
    ("significant_dates", "Significant dates (dd-mm-yyyy, dd/mm/yyyy, yyyy-mm-dd, etc.) (e.g., 31-12-2020)"),
    ("additional", "Additional custom words (e.g., projectx, codename)"),
]


def parse_csv_field(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def profile_wizard() -> Dict[str, List[str]]:
    profile: Dict[str, List[str]] = {}
    print("\nEnter values separated by commas. Leave blank to skip. Example input: item1, item2, item3. Type &&& then ENTER to go back.")
    idx = 0
    total = len(PROFILE_PROMPTS)
    while idx < total:
        key, label = PROFILE_PROMPTS[idx]
        try:
            raw = input(f"{label}: ").strip()
            if raw == BACK_TOKEN:
                raise Backtrack()
            profile[key] = parse_csv_field(raw) if raw else []
            idx += 1
        except Backtrack:
            if idx > 0:
                idx -= 1
                print("Going back to previous question.")
            else:
                print("Already at the first question.")
    return profile


def sanitize_path_segment(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]", "_", name.strip().lower().replace(" ", "_"))
    cleaned = cleaned.rstrip(" .")
    return cleaned or "subject"


def normalize_token(token: str) -> Optional[str]:
    cleaned = " ".join(token.strip().split())
    cleaned = cleaned.strip(".,;:!?\"'()[]{}")
    if not cleaned:
        return None
    ascii_form = (
        unicodedata.normalize("NFKD", cleaned)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    ascii_form = ascii_form.strip()
    if not ascii_form:
        return None
    return ascii_form


def toggle_case(value: str) -> str:
    return "".join(ch.lower() if ch.isupper() else ch.upper() for ch in value)


def case_variants(token: str, full: bool = True) -> Set[str]:
    base = token
    variants = {base.lower(), base.upper(), base.capitalize(), toggle_case(base)}
    if not full:
        variants = {base.lower(), base.capitalize()}
    collapsed = base.replace(" ", "")
    variants.add(collapsed.lower())
    variants.add(collapsed.capitalize())
    return {v for v in variants if v}


def parse_dates(values: Iterable[str]) -> Set[str]:
    date_forms: Set[str] = set()
    patterns = [
        r"^(?P<d>\d{2})[-/.](?P<m>\d{2})[-/.](?P<y>\d{4})$",
        r"^(?P<y>\d{4})[-/.](?P<m>\d{2})[-/.](?P<d>\d{2})$",
        r"^(?P<d>\d{2})[-/.](?P<m>\d{2})[-/.](?P<y>\d{2})$",
        r"^(?P<y>\d{2})[-/.](?P<m>\d{2})[-/.](?P<d>\d{2})$",
        r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{4})$",
        r"^(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})$",
        r"^(?P<d>\d{2})(?P<m>\d{2})(?P<y>\d{2})$",
        r"^(?P<y>\d{2})(?P<m>\d{2})(?P<d>\d{2})$",
    ]
    for raw in values:
        token = raw.strip()
        if not token:
            continue
        for pat in patterns:
            match = re.match(pat, token)
            if not match:
                continue
            parts = match.groupdict()
            d = parts.get("d")
            m = parts.get("m")
            y = parts.get("y")
            yy = y[-2:] if y else None
            if d and m:
                date_forms.update({d, m, d + m, m + d})
            if yy:
                date_forms.update({yy, (d or "") + (m or "") + yy, yy + (m or "") + (d or "")})
            if y:
                date_forms.add(y)
            if d and m and y:
                date_forms.update({d + m + y, y + m + d})
            break
    return {form for form in date_forms if form}


def leet_variants(word: str, max_changes: int = 2) -> Set[str]:
    mapping = {"a": ["@", "4"], "e": ["3"], "i": ["1"], "o": ["0"], "s": ["$", "5"], "t": ["7"]}
    lower_word = word.lower()
    positions = [(idx, mapping[ch]) for idx, ch in enumerate(lower_word) if ch in mapping]
    variants: Set[str] = {word}
    for changes in range(1, max_changes + 1):
        for combo in itertools.combinations(positions, changes):
            for replacements in itertools.product(*[opts for _, opts in combo]):
                chars = list(word)
                for (idx, _), val in zip(combo, replacements):
                    chars[idx] = val
                variants.add("".join(chars))
    return variants


def has_policy_requirements(candidate: str, policy: Policy) -> bool:
    if len(candidate) < policy.min_length or len(candidate) > policy.max_length:
        return False
    if policy.disallow_whitespace and any(ch.isspace() for ch in candidate):
        return False
    if policy.disallow_repeats and any(candidate[i] == candidate[i + 1] for i in range(len(candidate) - 1)):
        return False
    if policy.require_upper and not any(ch.isupper() for ch in candidate):
        return False
    if policy.require_lower and not any(ch.islower() for ch in candidate):
        return False
    if policy.require_digit and not any(ch.isdigit() for ch in candidate):
        return False
    if policy.require_special and not any(ch in "!@#-_." for ch in candidate):
        return False
    return True


def gather_tokens(profile: Dict[str, List[str]]) -> Tuple[List[str], Set[str], str]:
    tokens: List[str] = []
    date_inputs: List[str] = []
    subject = "subject"
    for key, values in profile.items():
        for value in values:
            normalized = normalize_token(value)
            if not normalized:
                continue
            if subject == "subject" and key.startswith("personal_"):
                subject = normalized
            if re.fullmatch(r"\d{2}[-/.]\d{2}[-/.]\d{2,4}|\d{8}", normalized) or re.fullmatch(r"\d{6}", normalized):
                date_inputs.append(normalized)
            tokens.append(normalized)
    date_forms = parse_dates(date_inputs)
    subject = sanitize_path_segment(subject)
    return tokens, date_forms, subject


def build_token_variants(tokens: List[str], transform_set: str) -> Set[str]:
    variants: Set[str] = set()
    full = transform_set in ("plus", "insane")
    for token in tokens:
        variants.update(case_variants(token, full=full))
    return variants


def generate_candidates(
    base_tokens: Set[str],
    date_forms: Set[str],
    policy: Policy,
    transform_set: str,
) -> Iterable[str]:
    full_cases = transform_set in ("plus", "insane")
    tokens = [t for t in base_tokens if len(t) <= policy.max_length]
    years = {d for d in date_forms if len(d) in (2, 4)}
    condensed_dates = {d for d in date_forms if len(d) in (4, 6, 8)}
    numeric_parts = set(NUMBER_PATTERNS) | years | condensed_dates
    basic_variants = set()
    for t in tokens:
        basic_variants.update(case_variants(t, full=False))
    # Single tokens and simple date/year attachments
    for t in (case_variants(tok, full=full_cases) for tok in tokens):
        for variant in t:
            yield variant
            for d in years | condensed_dates:
                candidate = variant + d
                if len(candidate) <= policy.max_length:
                    yield candidate
    if transform_set == "basic":
        return
    # Pairwise combinations and numeric patterns
    token_list = tokens
    for a, b in itertools.permutations(token_list, 2):
        for sep in SEPARATORS:
            combined = a + sep + b
            if len(combined) <= policy.max_length:
                for variant in case_variants(combined, full=full_cases):
                    yield variant
                for suffix in SUFFIXES + MIXED_SUFFIXES:
                    combo = combined + suffix
                    if len(combo) <= policy.max_length:
                        for variant in case_variants(combo, full=full_cases):
                            yield variant
    for t in token_list:
        reversed_token = t[::-1]
        if len(reversed_token) <= policy.max_length:
            for variant in case_variants(reversed_token, full=full_cases):
                yield variant
        for pattern in NUMBER_PATTERNS:
            candidate = t + pattern
            if len(candidate) <= policy.max_length:
                for variant in case_variants(candidate, full=full_cases):
                    yield variant
            for special in SPECIAL_CHARS:
                combo = t + pattern + special
                if len(combo) <= policy.max_length:
                    for variant in case_variants(combo, full=full_cases):
                        yield variant
                combo2 = t + special + pattern
                if len(combo2) <= policy.max_length:
                    for variant in case_variants(combo2, full=full_cases):
                        yield variant
        for suffix in SUFFIXES:
            candidate = t + suffix
            if len(candidate) <= policy.max_length:
                for variant in case_variants(candidate, full=full_cases):
                    yield variant
        for mixed in MIXED_SUFFIXES:
            candidate = t + mixed
            if len(candidate) <= policy.max_length:
                for variant in case_variants(candidate, full=full_cases):
                    yield variant
        for date in condensed_dates:
            combo = t + date
            if len(combo) <= policy.max_length:
                for variant in case_variants(combo, full=full_cases):
                    yield variant
        for special in SPECIAL_CHARS:
            with_special = t + date + special
            if len(with_special) <= policy.max_length:
                for variant in case_variants(with_special, full=full_cases):
                    yield variant
            with_special_front = t + special + date
            if len(with_special_front) <= policy.max_length:
                for variant in case_variants(with_special_front, full=full_cases):
                    yield variant
    # Mix words and numeric parts in all permutations of length 2
    pieces_plus = token_list + list(numeric_parts)
    for combo in itertools.permutations(pieces_plus, 2):
        for sep in SEPARATORS:
            combined = sep.join(combo)
            if len(combined) > policy.max_length:
                continue
            if any(c.isalpha() for c in combined):
                for variant in case_variants(combined, full=full_cases):
                    yield variant
            else:
                yield combined
    if transform_set != "insane":
        return
    capped_tokens = token_list[:25]
    capped_mix = pieces_plus[:30]
    for combo in itertools.permutations(capped_tokens, 3):
        for sep in SEPARATORS:
            candidate = sep.join(combo)
            if len(candidate) > policy.max_length:
                continue
            for variant in case_variants(candidate, full=True):
                yield variant
    for combo in itertools.permutations(capped_mix, 3):
        for sep in SEPARATORS:
            candidate = sep.join(combo)
            if len(candidate) > policy.max_length:
                continue
            if any(c.isalpha() for c in candidate):
                for variant in case_variants(candidate, full=True):
                    yield variant
            else:
                yield candidate


def write_candidates(
    candidates: Iterable[str],
    policy: Policy,
    leet_enabled: bool,
    preview: int,
    output_root: str,
    subject_dir: str,
    quiet: bool,
    exact_dedupe: bool,
) -> Tuple[int, int, int]:
    deduper = ExactDeduper() if exact_dedupe else BloomDeduper()
    processed = 0
    valid = 0
    files_created = 0
    writer = None
    lines_in_file = 0
    output_path = os.path.join(output_root, subject_dir)
    if preview == 0:
        os.makedirs(output_path, exist_ok=True)

    def open_file(idx: int):
        filename = f"wordlist_{idx:05d}.txt"
        return open(os.path.join(output_path, filename), "w", encoding="utf-8", newline="")

    try:
        for candidate in candidates:
            if len(candidate) < policy.min_length - 4:
                continue
            leet_options = [candidate]
            if leet_enabled:
                leet_options = list(leet_variants(candidate))
            for variant in leet_options:
                processed += 1
                if not has_policy_requirements(variant, policy):
                    continue
                if deduper.check_or_add(variant):
                    continue
                valid += 1
                if preview:
                    if valid <= preview:
                        print(variant)
                    if valid >= preview:
                        return processed, valid, files_created
                    continue
                if writer is None or lines_in_file >= FILE_LINE_LIMIT:
                    if writer:
                        writer.close()
                    files_created += 1
                    writer = open_file(files_created)
                    lines_in_file = 0
                writer.write(variant + "\n")
                lines_in_file += 1
        if writer:
            writer.close()
    except OSError as exc:
        if writer:
            writer.close()
        raise RuntimeError(f"Filesystem error: {exc}") from exc
    return processed, valid, files_created


def load_profile_from_json(stdin_data: str) -> Dict[str, List[str]]:
    try:
        loaded = json.loads(stdin_data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON input: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("Profile JSON must be an object.")

    def collect(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, dict):
            items: List[str] = []
            for _, val in value.items():
                items.extend(collect(val))
            return items
        return [str(value)]

    profile: Dict[str, List[str]] = {}
    for key, _ in PROFILE_PROMPTS:
        profile[key] = collect(loaded.get(key, []))
    # Include any additional stray keys
    for key, value in loaded.items():
        if key not in profile:
            profile[key] = collect(value)
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ethical CUPP-style wordlist generator.",
        add_help=True,
    )
    parser.add_argument("-cupp", "--cupp", action="store_true", help="Enable CUPP workflow.")
    parser.add_argument("-o", "--output", default="passwords", help='Root output directory (default: "passwords").')
    parser.add_argument("--min-length", type=int, help="Override minimum length policy.")
    parser.add_argument("--max-length", type=int, help="Override maximum length policy.")
    parser.add_argument("--no-leet", action="store_true", help="Disable leetspeak transformations.")
    parser.add_argument(
        "--transform-set",
        choices=TRANSFORM_CHOICES,
        default="plus",
        help="Transform intensity: basic | plus | insane (default: plus).",
    )
    parser.add_argument("--preview", type=int, default=PREVIEW_DEFAULT, help="Print the first N valid passwords and exit.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Non-interactive mode: skip banner/prompts, read JSON profile from stdin, then generate.",
    )
    parser.add_argument(
        "--exact-dedupe",
        action="store_true",
        help="Use exact deduplication (more memory) to avoid any false positives.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.quiet:
        print_banner()

    if not args.cupp:
        parser.print_help()
        sys.exit(0)

    min_length = args.min_length if args.min_length else DEFAULT_MIN_LENGTH
    max_length = args.max_length if args.max_length else DEFAULT_MAX_LENGTH
    if max_length < min_length:
        print("Error: maximum length cannot be smaller than minimum length.", file=sys.stderr)
        sys.exit(1)

    if args.quiet:
        stdin_data = sys.stdin.read()
        if not stdin_data.strip():
            print("Error: no JSON profile supplied on stdin for --quiet mode.", file=sys.stderr)
            sys.exit(1)
        try:
            profile = load_profile_from_json(stdin_data)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        policy = Policy(
            min_length=min_length,
            max_length=max_length,
            require_upper=True,
            require_lower=True,
            require_digit=True,
            require_special=True,
            disallow_whitespace=True,
            disallow_repeats=True,
        )
    else:
        policy = policy_wizard(args)
        profile = profile_wizard()

    tokens, date_forms, subject = gather_tokens(profile)
    if not tokens and not date_forms:
        print("No tokens collected after normalization. Nothing to generate.", file=sys.stderr)
        sys.exit(0)

    base_tokens = build_token_variants(tokens, args.transform_set)
    if not base_tokens:
        print("No usable tokens after normalization. Nothing to generate.", file=sys.stderr)
        sys.exit(0)

    start_time = time.time()
    candidates = generate_candidates(base_tokens, date_forms, policy, args.transform_set)
    try:
        processed, valid, files_created = write_candidates(
            candidates=candidates,
            policy=policy,
            leet_enabled=not args.no_leet,
            preview=args.preview,
            output_root=args.output,
            subject_dir=subject,
            quiet=args.quiet,
            exact_dedupe=args.exact_dedupe,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    elapsed = time.time() - start_time

    if not args.quiet:
        print("\nGeneration complete:")
        print(f"- total candidates processed: {processed}")
        print(f"- total valid: {valid}")
        print(f"- files created: {files_created}")
        print(f"- output directory: {os.path.join(args.output, subject)}")
        print(f"- elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
