import sys
from pathlib import Path


def parse_frontmatter(text: str):
    lines = text.splitlines()
    if not lines:
        return None
    if lines[0].strip() != '---':
        return None
    fm_lines = []
    for ln in lines[1:]:
        if ln.strip() == '---':
            break
        fm_lines.append(ln)
    data = {}
    for ln in fm_lines:
        if ':' in ln:
            k, v = ln.split(':', 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def validate_skill(path: Path):
    if not path.exists():
        print(f'MISSING: {path}')
        return 2
    text = path.read_text(encoding='utf-8')
    fm = parse_frontmatter(text)
    errors = []
    if not fm:
        errors.append('Frontmatter YAML block (---) not found at top')
    else:
        for required in ('name', 'description'):
            if required not in fm or not fm[required]:
                errors.append(f"Missing frontmatter field: {required}")

    # simple content checks
    if '## When to Use This Skill' not in text:
        errors.append('Missing section: ## When to Use This Skill')

    if errors:
        print('VALIDATION FAILED')
        for e in errors:
            print('- ' + e)
        return 1

    print('SKILL OK — frontmatter and basic sections present')
    print('Parsed frontmatter:')
    for k, v in fm.items():
        print(f' - {k}: {v}')
    return 0


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[3]
    skill_path = repo_root / '.claude' / 'skills' / 'file-organizer' / 'SKILL.md'
    code = validate_skill(skill_path)
    sys.exit(code)
