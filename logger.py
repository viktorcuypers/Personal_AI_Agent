from datetime import datetime
from pathlib import Path

UNANSWERED_LOG = Path("data/unanswered.txt")
UNANSWERED_LOG.parent.mkdir(parents=True, exist_ok=True)

def log_unanswered(question: str, sources: list[str], reason: str = ""):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    with UNANSWERED_LOG.open("a", encoding="utf-8") as f:
        f.write("=====================================\n")
        f.write(f"Time: {timestamp}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Sources used: {', '.join(sources) if sources else 'None'}\n")
        if reason:
            f.write(f"Reason: {reason}\n")
        f.write("\n")