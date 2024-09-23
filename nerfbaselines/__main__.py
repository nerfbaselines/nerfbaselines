import shutil
from .cli import main

if __name__ == "__main__":
    main(max_content_width=min(120, shutil.get_terminal_size().columns))
