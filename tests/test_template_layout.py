from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TemplateLayoutTests(unittest.TestCase):
    def test_default_config_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "configs" / "default.yaml").is_file())

    def test_train_script_exists(self) -> None:
        self.assertTrue((REPO_ROOT / "scripts" / "train.py").is_file())


if __name__ == "__main__":
    unittest.main()
