import importlib.util
import sys
from pathlib import Path

module_path = Path(__file__).with_name("2_8_agent_ui_api.py")
src_dir = str(module_path.parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
spec = importlib.util.spec_from_file_location("chapter_2_8", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

app = module.app
