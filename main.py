"""
main.py - Multi-Agent RAG System for Course Q&A

Architecture (from Paper 1 - Agentic RAG, Section 4.2):
  - Orchestrator Agent: Plans, delegates to sub-agents, synthesizes final answer
  - Retriever Agent: Searches local course documents via ChromaDB
  - Web Search Agent: Searches the web via DuckDuckGo for supplementary info

Framework inspiration (from Paper 2 - Orchestral AI):
  - Provider-agnostic design (Ollama for local LLM)
  - Tool-based agent architecture with clear separation
  - Synchronous execution for deterministic behavior
  - Subagent delegation pattern (hierarchical orchestration)

Usage:
    1. First run: python ingest.py   (loads documents into ChromaDB)
    2. Then run:  python main.py      (starts the Q&A system)
"""

import json
import ollama
from tools.retriever import search_course_docs
from tools.web_search import search_web


# ============================================================
# Tool Definitions (Orchestral AI style - clear schema per tool)
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_course_docs",
            "description": (
                "Search local course documents (notes, slides, textbooks) "
                "stored in the vector database. Use this FIRST for any "
                "course-related question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query about course material"
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for supplementary information. Use this "
                "when local course documents don't fully answer the question "
                "or when current/external information is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 3)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Map tool names to actual functions
TOOL_FUNCTIONS = {
    "search_course_docs": search_course_docs,
    "search_web": search_web,
}


# ============================================================
# Agent System Prompts
# ============================================================

ORCHESTRATOR_PROMPT = """You are an intelligent Course Q&A Assistant built using a multi-agent RAG architecture.

Your role is to help students answer questions about their course material accurately and clearly.

WORKFLOW (Multi-Agent RAG pattern from Agentic RAG research):
1. RETRIEVE: Always start by searching local course documents using 'search_course_docs'
2. SUPPLEMENT: If the local docs don't fully answer the question, also use 'search_web'
3. SYNTHESIZE: Combine information from both sources into a clear, student-friendly answer

RULES:
- Always cite your sources (indicate if info came from course docs or web)
- If course docs have the answer, prioritize that over web results
- Keep answers clear, concise, and educational
- If you're unsure, say so honestly
- Format answers in a way that helps students learn (use examples when helpful)

You have access to:
- search_course_docs: Searches local course notes/documents in the vector store
- search_web: Searches the web via DuckDuckGo for supplementary information
"""


# ============================================================
# Orchestrator Agent (Orchestral AI pattern: Agent + Tools + Loop)
# ============================================================

class OrchestratorAgent:
    """
    Multi-agent orchestrator that manages the RAG pipeline.

    Follows Paper 2 (Orchestral AI) design:
    - Single Agent object as the orchestrator
    - Tools registered with clear schemas
    - Synchronous execution loop
    - Conversation context maintained across turns

    Implements Paper 1 (Agentic RAG) Section 4.2:
    - Multi-agent collaboration via tool-based delegation
    - Retriever agent (ChromaDB tool)
    - Web search agent (DuckDuckGo tool)
    - Orchestrator synthesizes results
    """

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.conversation_history = []
        self.system_prompt = ORCHESTRATOR_PROMPT

    def run(self, user_query: str, max_iterations: int = 5) -> str:
        """
        Process a user query through the multi-agent RAG pipeline.

        Args:
            user_query: The student's question
            max_iterations: Max tool-calling rounds (safety limit)

        Returns:
            The final synthesized answer
        """
        print(f"\n{'━' * 70}")
        print(f"  📝 QUERY: {user_query}")
        print(f"{'━' * 70}")

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        # Add conversation history for context
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_query})

        iteration = 0
        all_sources = []  # Track all sources used

        while iteration < max_iterations:
            iteration += 1

            # Call Ollama with tools
            print(f"\n  {'─' * 66}")
            print(f"  🤖 ORCHESTRATOR - Iteration {iteration}/{max_iterations}")
            print(f"  {'─' * 66}")

            # Show what the orchestrator is thinking about
            if iteration == 1:
                print(f"  📋 PLAN: Analyzing query and deciding which agents to invoke...")
            else:
                print(f"  📋 PLAN: Reviewing retrieved information, deciding next step...")

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS,
            )

            assistant_message = response["message"]
            messages.append(assistant_message)

            # Show if the model had any intermediate reasoning
            if assistant_message.get("content"):
                thinking = assistant_message["content"].strip()
                if thinking:
                    print(f"\n  💭 ORCHESTRATOR THINKING:")
                    for line in thinking.split('\n'):
                        print(f"     {line}")

            # Check if the model wants to call tools
            if not assistant_message.get("tool_calls"):
                # No more tool calls - we have the final answer
                final_answer = assistant_message.get("content", "")

                print(f"\n  {'━' * 66}")
                print(f"  ✅ SYNTHESIS COMPLETE")
                print(f"  {'━' * 66}")

                if all_sources:
                    print(f"  📚 SOURCES USED:")
                    for src in all_sources:
                        print(f"     • {src}")

                print(f"  🔄 TOTAL ITERATIONS: {iteration}")
                print(f"  {'━' * 66}")

                # Save to conversation history
                self.conversation_history.append(
                    {"role": "user", "content": user_query}
                )
                self.conversation_history.append(
                    {"role": "assistant", "content": final_answer}
                )

                return final_answer

            # Execute tool calls (sub-agent delegation)
            for tool_call in assistant_message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]

                # Identify which agent is being called
                if func_name == "search_course_docs":
                    agent_label = "📖 RETRIEVER AGENT (ChromaDB)"
                    source_label = "Course Docs"
                elif func_name == "search_web":
                    agent_label = "🌐 WEB SEARCH AGENT (Wikipedia)"
                    source_label = "Wikipedia"
                else:
                    agent_label = f"🔧 TOOL: {func_name}"
                    source_label = func_name

                print(f"\n  {agent_label}")
                print(f"  ├─ Function: {func_name}")
                print(f"  ├─ Arguments: {json.dumps(func_args)}")

                # Execute the tool
                if func_name in TOOL_FUNCTIONS:
                    result = TOOL_FUNCTIONS[func_name](**func_args)
                else:
                    result = f"Unknown tool: {func_name}"

                # Truncate very long results to avoid context overflow
                if len(result) > 3000:
                    result = result[:3000] + "\n...[truncated]"

                # Show retrieved content preview
                print(f"  ├─ Result: {len(result)} characters retrieved")
                print(f"  └─ Preview:")

                # Show first few lines of the result
                preview_lines = result.split('\n')
                for i, line in enumerate(preview_lines[:12]):
                    line_trimmed = line.strip()
                    if line_trimmed:
                        if len(line_trimmed) > 90:
                            line_trimmed = line_trimmed[:90] + "..."
                        print(f"     {line_trimmed}")
                if len(preview_lines) > 12:
                    print(f"     ... ({len(preview_lines) - 12} more lines)")

                # Track sources
                query_used = func_args.get("query", "N/A")
                all_sources.append(f"{source_label} → query: \"{query_used}\"")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "content": result,
                })

        return "I reached the maximum number of reasoning steps. Please try rephrasing your question."

    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")


# ============================================================
# Interactive CLI
# ============================================================

def print_banner():
    print()
    print("━" * 70)
    print("  🎓 Multi-Agent RAG - Course Q&A System (Verbose Mode)")
    print("  Paper 1: Agentic RAG (Section 4.2 - Multi-Agent)")
    print("  Paper 2: Orchestral AI (Agent Orchestration)")
    print("━" * 70)
    print("  🤖 LLM:       Ollama (local/cloud)")
    print("  📖 Retriever:  ChromaDB vector store")
    print("  🌐 Web Search: Wikipedia API")
    print("─" * 70)
    print("  Pipeline: Query → Orchestrator → [Retriever + Web] → Synthesis")
    print("─" * 70)
    print("  Commands:")
    print("    Type your question and press Enter")
    print("    'reset' - Clear conversation history")
    print("    'quit'  - Exit the program")
    print("━" * 70)


def main():
    print_banner()

    # Check if ChromaDB has been populated
    import os
    chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    if not os.path.exists(chroma_dir):
        print("\n  [!] ChromaDB not found. Running ingestion first...")
        print("  [!] Run: python ingest.py\n")
        return

    # Check if Ollama is running
    try:
        ollama.list()
    except Exception:
        print("\n  [!] Cannot connect to Ollama.")
        print("  [!] Make sure Ollama is running: ollama serve")
        print("  [!] And pull a model: ollama pull llama3.2\n")
        return

    # Initialize the orchestrator agent
    agent = OrchestratorAgent(model="gpt-oss:120b-cloud")
    print("\n  Ready! Ask me anything about your courses.\n")

    while True:
        try:
            user_input = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n  Goodbye!")
            break

        if user_input.lower() == "reset":
            agent.reset()
            continue

        # Run the multi-agent pipeline
        answer = agent.run(user_input)
        print(f"\n  Assistant: {answer}\n")


if __name__ == "__main__":
    main()