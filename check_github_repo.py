# check_github_repo.py
import os
import json
from pathlib import Path
import sys

# ─── تنظیمات ──────────────────────────────────────────────────────
REPO_ROOT = Path(r"C:\ptbxl\ptbxl-ensemble-myocardial-infarction")  # مسیر ریپوزیتوری محلی
PROJECT_ROOT = Path(r"C:\ptbxl")  # مسیر پروژه اصلی

# ─── فایل‌ها و پوشه‌های مورد نیاز ─────────────────────────────
REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "hyperparameters.json",
    "run_pipeline.py",
    ".gitignore",
]

REQUIRED_DIRS = [
    "src",
    "results",
]

REQUIRED_RESULTS = [
    "paper1",
    "paper2",
    "student",
    "auto_teacher",
]

# ─── فایل‌هایی که نباید در گیت‌هاب باشند ──────────────────────
FORBIDDEN_EXTENSIONS = [".pt", ".pth", ".onnx"]
FORBIDDEN_DIRS = [
    "data",
    "project",
    "logs",
    "venv",
    "ecg_env",
    "submission",
    "__pycache__",
]

MAX_FILE_SIZE_MB = 10  # حداکثر حجم مجاز هر فایل (مگابایت)


def print_header(text):
    print("\n" + "=" * 60)
    print(f"🔍 {text}")
    print("=" * 60)


def check_required_files():
    """بررسی وجود فایل‌های اجباری"""
    print_header("بررسی فایل‌های اجباری")
    all_ok = True
    for f in REQUIRED_FILES:
        path = REPO_ROOT / f
        if path.exists():
            size = path.stat().st_size / 1024  # KB
            print(f"   ✅ {f} ({(size/1024):.2f} MB)")
        else:
            print(f"   ❌ {f} - وجود ندارد")
            all_ok = False
    return all_ok


def check_required_dirs():
    """بررسی وجود پوشه‌های اجباری"""
    print_header("بررسی پوشه‌های اجباری")
    all_ok = True
    for d in REQUIRED_DIRS:
        path = REPO_ROOT / d
        if path.exists() and path.is_dir():
            count = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"   ✅ {d}/ ({count} فایل)")
        else:
            print(f"   ❌ {d}/ - وجود ندارد")
            all_ok = False

    # بررسی زیرپوشه‌های results
    results_path = REPO_ROOT / "results"
    if results_path.exists():
        for sub in REQUIRED_RESULTS:
            sub_path = results_path / sub
            if sub_path.exists() and sub_path.is_dir():
                count = sum(1 for _ in sub_path.rglob("*") if _.is_file())
                print(f"      ✅ results/{sub}/ ({count} فایل)")
            else:
                print(f"      ❌ results/{sub}/ - وجود ندارد")
                all_ok = False
    return all_ok


def check_forbidden_files():
    """بررسی فایل‌های ممنوع (حجم بالا)"""
    print_header("بررسی فایل‌های ممنوع (حجم بالا)")

    large_files = []
    total_size_mb = 0

    for ext in FORBIDDEN_EXTENSIONS:
        for f in REPO_ROOT.rglob(f"*{ext}"):
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            large_files.append((f.relative_to(REPO_ROOT), size_mb))

    if large_files:
        print("   ⚠️  فایل‌های بزرگ (بهتر است در گیت‌هاب نباشند):")
        for name, size in large_files:
            print(f"      - {name} ({size:.2f} MB)")
        print(f"\n   مجموع حجم فایل‌های بزرگ: {total_size_mb:.2f} MB")
        if total_size_mb > 100:
            print("   ❌ حجم کل فایل‌های بزرگ بیش از 100 MB است!")
            return False
        else:
            print("   ✅ حجم کل فایل‌های بزرگ کمتر از 100 MB است.")
    else:
        print("   ✅ هیچ فایل بزرگ (pt/pth/onnx) یافت نشد.")

    return True


def check_forbidden_dirs():
    """بررسی پوشه‌های ممنوع"""
    print_header("بررسی پوشه‌های غیرضروری")
    found = []
    for d in FORBIDDEN_DIRS:
        path = REPO_ROOT / d
        if path.exists() and path.is_dir():
            count = sum(1 for _ in path.rglob("*") if _.is_file())
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
            found.append((d, count, size))

    if found:
        print("   ⚠️  پوشه‌های زیر در ریپوزیتوری وجود دارند (بهتر است حذف شوند):")
        for name, count, size in found:
            print(f"      - {name}/ ({count} فایل, {size:.2f} MB)")
        return False
    else:
        print("   ✅ هیچ پوشه‌ی غیرضروری یافت نشد.")
        return True


def check_gitignore():
    """بررسی فایل .gitignore"""
    print_header("بررسی .gitignore")
    path = REPO_ROOT / ".gitignore"
    if path.exists():
        content = path.read_text(encoding="utf-8")
        patterns = ["*.pt", "*.pth", "data/", "project/", "venv/", "__pycache__/"]
        missing = [p for p in patterns if p not in content]
        if missing:
            print("   ⚠️  الگوهای زیر در .gitignore وجود ندارند:")
            for m in missing:
                print(f"      - {m}")
            return False
        else:
            print("   ✅ .gitignore کامل است.")
            return True
    else:
        print("   ❌ فایل .gitignore وجود ندارد.")
        return False


def check_overall_size():
    """بررسی حجم کل ریپوزیتوری"""
    print_header("بررسی حجم کل ریپوزیتوری")
    total_size = sum(f.stat().st_size for f in REPO_ROOT.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"   حجم کل: {total_size:.2f} MB")
    if total_size > 100:
        print("   ⚠️  حجم کل بیش از 100 MB است. ممکن است گیت‌هاب محدودیت ایجاد کند.")
    else:
        print("   ✅ حجم کل مناسب است (< 100 MB).")
    return total_size


def main():
    print_header("بررسی محتوای ریپوزیتوری برای آپلود در گیت‌هاب")
    print(f"📁 مسیر ریپوزیتوری: {REPO_ROOT}\n")

    if not REPO_ROOT.exists():
        print(f"❌ مسیر ریپوزیتوری وجود ندارد: {REPO_ROOT}")
        print("   لطفاً مسیر صحیح را در اسکریپت تنظیم کنید.")
        sys.exit(1)

    results = {
        "required_files": check_required_files(),
        "required_dirs": check_required_dirs(),
        "forbidden_files": check_forbidden_files(),
        "forbidden_dirs": check_forbidden_dirs(),
        "gitignore": check_gitignore(),
    }

    total_size_mb = check_overall_size()

    print("\n" + "=" * 60)
    print("📋 گزارش نهایی")
    print("=" * 60)

    for key, value in results.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}")

    all_ok = all(results.values())

    if all_ok:
        print("\n✅ همه موارد تأیید شد. ریپوزیتوری برای آپلود آماده است.")
    else:
        print("\n⚠️ برخی موارد نیاز به اصلاح دارند. لطفاً پیام‌های بالا را بررسی کنید.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()