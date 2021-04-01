from pathlib import Path
import subprocess
import sys

if __name__ != '__main__':
    raise ImportError(__name__ + ' is not meant be imported')

self = Path(__file__)
cwd = self.parent

for script in cwd.glob('*.py'):
    if self == script:
        # Don't call yourself!
        continue
    print('Running', script, '...')
    args = [sys.executable, str(script.relative_to(cwd))] + sys.argv[1:]
    result = subprocess.run(args, cwd=str(cwd))
    if result.returncode:
        print('Error running', script, file=sys.stderr)
        sys.exit(result.returncode)
        