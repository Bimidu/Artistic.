#!/usr/bin/env python
"""Fix Python 3.8 compatibility issues - replace | syntax with Union"""

import re
from pathlib import Path

files_to_fix = [
    "src/models/model_trainer.py",
    "src/models/pragmatic_conversational/model_trainer.py",
    "src/preprocessing/data_validator.py",
    "src/preprocessing/preprocessor.py",
]

def fix_file(file_path):
    """Fix union type syntax in a file"""
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        return False

    content = path.read_text()
    original_content = content

    # Check if Union is imported
    has_union_import = 'from typing import' in content and 'Union' in content

    # Add Union to imports if needed
    if not has_union_import and 'from typing import' in content:
        # Find the typing import line
        import_match = re.search(r'from typing import ([^\n]+)', content)
        if import_match:
            imports = import_match.group(1)
            if 'Union' not in imports:
                # Add Union to imports
                new_imports = imports.rstrip() + ', Union'
                content = content.replace(
                    f'from typing import {imports}',
                    f'from typing import {new_imports}'
                )
                print(f"  Added Union to imports in {file_path}")

    # Replace | syntax with Union
    # Pattern: type1 | type2
    pattern = r'\b(\w+)\s*\|\s*(\w+)\b'

    def replace_union(match):
        type1 = match.group(1)
        type2 = match.group(2)
        return f'Union[{type1}, {type2}]'

    content = re.sub(pattern, replace_union, content)

    if content != original_content:
        path.write_text(content)
        print(f"✓ Fixed {file_path}")
        return True
    else:
        print(f"  No changes needed for {file_path}")
        return False

print("Fixing Python 3.8 compatibility issues...")
print("="*70)

fixed_count = 0
for file_path in files_to_fix:
    if fix_file(file_path):
        fixed_count += 1

print("="*70)
print(f"Fixed {fixed_count} files")
