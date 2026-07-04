import argparse
import os

from support_agent.agent import SupportAgent
from support_agent.config import get_settings
from support_agent.human_interface import HumanAgentInterface
from support_agent.state_store import StateStore


def build_agent(project_root: str) -> SupportAgent:
    settings = get_settings()
    state_store = StateStore(db_path=settings.database_path)
    agent = SupportAgent(
        state_store=state_store,
        human_interface=HumanAgentInterface(state_store),
    )
    data_dir = os.path.join(project_root, "data")
    doc_paths = [
        os.path.join(data_dir, filename)
        for filename in os.listdir(data_dir)
        if filename.endswith(".md")
    ]
    print("Ingesting docs:", doc_paths)
    agent.ingest(doc_paths)
    return agent


def demo(agent: SupportAgent) -> None:
    print("Demo mode. Enter a support question, or type 'exit' to quit.")
    history = []

    while True:
        question = input("User: ").strip()
        if not question:
            continue
        if question.lower() == "exit":
            break

        answer = agent.answer(question, user_id="cli-user")
        print("Agent:", answer)
        history.append((question, answer))

    if history:
        print("Conversation ended.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(__file__)
    agent = build_agent(project_root)
    if args.demo:
        demo(agent)
    else:
        print("Run with --demo to start interactive demo")
