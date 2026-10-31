"""Audit git-tracked files for sensitive information, absolute paths, and secrets."""

import subprocess
import re
from pathlib import Path

def audit_tracked():
    out = subprocess.check_output(['git', 'ls-files'], text=True, encoding='utf-8')
    tracked_files = [f for f in out.strip().splitlines() if not f.endswith(('.jpg', '.png'))]

    patterns = {
        'marwan_path': re.compile(r'C:[/\\]Users[/\\]MARWAN', re.IGNORECASE),
        'truckentry_path': re.compile(r'D:[/\\]TruckEntry', re.IGNORECASE),
        'secrets': re.compile(r'(api[_\-]?key|secret[_\-]?key|private[_\-]?key|bearer\s+[a-zA-Z0-9_\-\.]{20,}|password\s*=\s*["\'][^"\']{4,})', re.IGNORECASE),
    }

    hits = {k: [] for k in patterns}

    for tf in tracked_files:
        p = Path(tf)
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        for name, pat in patterns.items():
            matches = pat.findall(content)
            if matches:
                hits[name].append((tf, len(matches)))

    print('=== TRACKED FILE SENSITIVITY AUDIT ===')
    for name, file_list in hits.items():
        print(f'\nPattern [{name}]: {len(file_list)} files matched')
        for f, count in file_list:
            print(f'   {f} ({count} occurrences)')

if __name__ == '__main__':
    audit_tracked()
