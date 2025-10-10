import sys
from pathlib import Path

# Add z3-agent to Python path
Z3_PATH = Path(__file__).parent.parent / "z3_agent"
if Z3_PATH.exists():
    sys.path.insert(0, str(Z3_PATH))
    print(f"z3-agent path added: {Z3_PATH}")
else:
    raise FileNotFoundError(f"z3-agent not found at {Z3_PATH}")