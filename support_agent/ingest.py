import os

from support_agent.agent import SupportAgent


def run_ingest() -> None:
    project_root = os.path.dirname(__file__)
    agent = SupportAgent()
    data_dir = os.path.join(project_root, "data")
    doc_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]
    agent.ingest(doc_paths)
    print(f"Ingest complete: {len(agent.chunks)} chunks ready")


if __name__ == "__main__":
    run_ingest()
