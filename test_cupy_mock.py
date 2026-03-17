import sys
from unittest.mock import MagicMock
sys.modules['cupy'] = MagicMock()
import core_transformer
print("Mocked cupy imported successfully!")
