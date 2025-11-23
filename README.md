# OCTO-CUPP – Advanced Password Profiler & Wordlist Generator

OCTO-CUPP is a **next-generation CUPP-style password profiling tool**, built for speed, modularity, and OSINT-driven wordlist creation.  
You provide profile data → the tool cooks thousands/millions of candidates → outputs clean wordlists split into **100,000-line chunks**.

No personal identifiers included.  
Interactive wizard + JSON automation.  
Straightforward. Ethical. Fast.

---

## 🚀 Features

- **CUPP workflow** enabled with `--cupp`
- **Colored ASCII banner**
- **Interactive policy wizard** (uppercase / lowercase / digits / specials / length / whitespace / repeats)
- **Bloom filter** (fast dedupe) + **Exact dedupe** (perfect, more RAM)
- **Transform sets:** `basic`, `plus`, `insane`
- **Leetspeak generator** (`a→4`, `e→3`, `o→0`, etc.)
- **Date parser** (01-01-2001 → 01012001 → 0101 → 01 → 2001)
- **Case variants** (lower, UPPER, Capitalized, tOgGlE)
- **Profile wizard** with 35+ OSINT fields:
  personal information, family, ex-partners, pets, cars, teams, usernames, slang, significant dates, etc.
- **Quiet mode** using JSON via stdin:
  `cat profile.json | python3 main.py --quiet --cupp`
- **Automatic directory structure per target**:
  `passwords/<subject>/wordlist_00001.txt`

---

## 📦 Installation

```bash
git clone https://github.com/h050sh/octo-cupp-profiler.git
cd octo-cupp-profiler
python3 main.py --help
```

---

## 🧪 Interactive Mode

Start the full wizard + banner:

```bash
python3 main.py --cupp
```

You will be asked:
- Policy questions (Y/n)
- Profile questions (comma-separated values)

Output goes to:

```
passwords/<subject>/
```

---

## 🤖 Quiet Mode (Automation / Scripting)

```bash
cat profile.json | python3 main.py --quiet --cupp --min-length 8 --max-length 20
```

Example `profile.json`:

```json
{
  "personal_full_names": ["john doe"],
  "favorites_games": ["minecraft"],
  "significant_dates": ["01-01-2001"]
}
```

---

## 🔍 Preview the First N Passwords

```bash
python3 main.py --cupp --preview 50
```

---

## 📁 Output Structure

Each file contains max 100,000 lines:

```
passwords/
└── john_doe/
    ├── wordlist_00001.txt
    ├── wordlist_00002.txt
    └── ...
```

---

## ⚙️ Transform Modes

Mode | Description
-----|-------------
**basic** | simple combinations only  
**plus** (default) | extended combinations, suffixes, special characters  
**insane** | full permutations (2–3 word combos), all variants, maximum expansion  

---

## ⚡ Performance

- **Bloom filter** → low-memory dedupe  
- **Exact dedupe** → perfect dedupe, more RAM needed  

Use exact dedupe:

```bash
python3 main.py --cupp --exact-dedupe
```

---

## 🎨 Banner Preview

```
  ____ _   _ ____  ____     _____           _        ____           _      
 / ___| | | |  _ \|  _ \   |_   _|__   ___ | |___   / ___|___ _ __ | | ___ 
| |   | | | | |_) | |_) |____| |/ _ \ / _ \| / __| | |   / _ \ '_ \| |/ _ \
| |___| |_| |  __/|  _ <_____| | (_) | (_) | \__ \ | |__|  __/ |_) | |  __/
 \____|\___/|_|   |_| \_\    |_|\___/ \___/|_|___/  \____\___| .__/|_|\___|
                                                             |_|             
```

---

## ⚠️ Legal Disclaimer

This tool is for **ethical security testing**, OSINT research, and educational purposes only.  
Use it **only** on systems where you have **explicit permission**.  
You are responsible for your own actions.

---

## ⭐ Support

If the project helps you, drop a ⭐ on the repo.
