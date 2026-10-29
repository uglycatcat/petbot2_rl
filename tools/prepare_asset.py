"""Run the read-only-model derived asset builder as a guarded script."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from upright.prepare import main
main()
