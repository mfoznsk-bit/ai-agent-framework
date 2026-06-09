import os
import json
from groq import Groq

# =========================================================
# ARCHITECTURE OVERVIEW — Read this first
# =========================================================
#      well all this is for handling the big complex question
# This is a multi-stage LLM orchestration pattern:
#
#   [User Goal]
#       │
#       ▼
#   Stage 1: PLANNER
#   Ask the LLM to decompose the goal into 3-5 subtasks.
#   Returns a JSON list — structured output we can loop over.
#       │
#       ▼
#   Stage 2: EXECUTOR (loop)
#   For each subtask, send it to the LLM for completion.
#   Each call is independent — the model focuses on one thing.
#       │
#       ▼
#   Stage 3: SYNTHESIZER
#   Feed all subtask results back to the LLM.
#   Ask it to combine them into a coherent final summary.
#       │
#       ▼
#   [Final Output]
#
# WHY THIS PATTERN?
# A single prompt like "do this complex business task" gives vague output.
# Breaking it into plan → execute → synthesize forces structured thinking
# and produces far more detailed, usable results.
# This is called "decomposition + orchestration" in agentic AI design.

# =========================================================
# SETUP
# =========================================================

client = Groq(api_key=os.getenv(""))

# 70B model — essential for structured JSON output and reasoning quality.
# Never use 8B models for orchestration tasks (they hallucinate JSON structure).
MODEL = "llama-3.3-70b-versatile"

# =========================================================
# HELPER: Single LLM call with clean error handling
# =========================================================

def call_llm(messages: list, label: str = "LLM call") -> str:
    """
    Makes one API call to Groq and returns the text response.

    Centralizing the API call here means:
    - All error handling lives in one place
    - Easy to add logging, retries, or token counting later
    - The 3 pipeline stages stay clean and readable

    messages: full conversation list (system + user + any history)
    label:    a name for this call, used in error messages
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.7
            # temperature=0.7 is a good balance:
            # 0.0 = deterministic, robotic
            # 1.0 = creative, unpredictable
            # 0.7 = professional, slightly varied
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  [ERROR in {label}]: {e}")
        return ""


# =========================================================
# STAGE 1: PLANNER
# =========================================================

def plan_subtasks(goal: str) -> list[dict]:
    """
    Takes a high-level business goal and asks the LLM to break it
    into 3-5 concrete, actionable subtasks.

    Returns a list of dicts, each with:
      - "id":          int, 1-based index
      - "title":       short name for the subtask
      - "description": what exactly needs to be done

    WHY RETURN JSON?
    Because we need to programmatically loop over the subtasks.
    Natural language output like "First, do X. Then, do Y." is
    hard to parse reliably. JSON gives us a clean iterable structure.

    WHY A SEPARATE SYSTEM PROMPT FOR EACH STAGE?
    Each stage has a different job. The planner needs to think like
    a project manager. The executor needs to think like a specialist.
    The synthesizer needs to think like a writer. Separate prompts
    = better role clarity = better output.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 1: PLANNING")
    print(f"{'='*60}")
    print(f"Goal: {goal}")
    print("Breaking into subtasks...")

    messages = [
        {
            "role": "system",
            "content": """You are a strategic business planner.
Your job is to decompose complex business goals into clear, actionable subtasks.

RULES:
- Break the goal into 3 to 5 subtasks. No more, no less.
- Each subtask must be concrete and independently executable.
- Return ONLY a valid JSON array. No explanation, no markdown, no code fences.
- Each item in the array must have exactly these three fields:
  {
    "id": <integer, starting from 1>,
    "title": "<short name, max 6 words>",
    "description": "<specific actionable instruction, 1-2 sentences>"
  }

Example output format (do not copy this content, just the structure):
[
  {"id": 1, "title": "Market Research", "description": "Identify the top 3 competitors..."},
  {"id": 2, "title": "Target Audience", "description": "Define the primary customer segment..."}
]"""
        },
        {
            "role": "user",
            "content": f"Decompose this business goal into 3-5 subtasks:\n\n{goal}"
        }
    ]

    raw_response = call_llm(messages, label="Planner")

    # ── Parse the JSON response ──────────────────────────────
    # The model might wrap output in ```json ... ``` even when told not to.
    # We strip those fences defensively before parsing.
    cleaned = raw_response.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        # Remove closing fence
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        subtasks = json.loads(cleaned)
        if not isinstance(subtasks, list):
            raise ValueError("Response is not a JSON array")

        print(f"\nSubtasks identified ({len(subtasks)}):")
        for task in subtasks:
            print(f"  [{task['id']}] {task['title']}")
            print(f"       → {task['description']}")

        return subtasks

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"  [ERROR] Failed to parse subtask JSON: {e}")
        print(f"  Raw response was:\n{raw_response}")
        # Graceful fallback: wrap the whole response as one subtask
        # so the pipeline doesn't crash — it degrades gracefully
        return [{"id": 1, "title": "Full Task", "description": goal}]


# =========================================================
# STAGE 2: EXECUTOR
# =========================================================

def execute_subtask(subtask: dict, goal: str, task_index: int, total_tasks: int) -> str:
    """
    Takes a single subtask and asks the LLM to complete it in detail.

    Each execution call:
    - Has full awareness of the original goal (for context)
    - Knows its position in the sequence (for narrative flow)
    - Focuses ONLY on its specific subtask (for depth)

    WHY NOT SEND ALL SUBTASKS AT ONCE?
    Focused prompts → focused, detailed answers.
    When you say "do 5 things", the model gives 20% effort to each.
    When you say "do this one thing, do it well", you get 100%.

    Returns the text output for this subtask.
    """
    print(f"\n{'─'*60}")
    print(f"STAGE 2: EXECUTING [{task_index}/{total_tasks}] — {subtask['title']}")
    print(f"{'─'*60}")

    messages = [
        {
            "role": "system",
            "content": f"""You are a business analyst and expert consultant.
You are working on a large business project. Your job right now is to complete
ONE specific subtask from the project plan.

Overall project goal: {goal}

Instructions:
- Focus entirely on the assigned subtask — do not address other parts of the project.
- Be specific, practical, and detailed.
- Write in clear professional English.
- Provide actionable content, not vague advice.
- Length: 2-4 paragraphs or a structured list — whatever fits the task best."""
        },
        {
            "role": "user",
            "content": (
                f"Complete this subtask (task {task_index} of {total_tasks}):\n\n"
                f"Title: {subtask['title']}\n"
                f"Task: {subtask['description']}"
            )
        }
    ]

    result = call_llm(messages, label=f"Executor [{subtask['title']}]")

    if result:
        # Preview the first 120 chars so the terminal isn't overwhelming
        preview = result[:120].replace("\n", " ")
        print(f"  Output preview: {preview}...")
    else:
        print("  [WARNING] Empty result for this subtask")
        result = f"(No output generated for: {subtask['title']})"

    return result


# =========================================================
# STAGE 3: SYNTHESIZER
# =========================================================

def synthesize_results(goal: str, subtasks: list[dict], results: list[str]) -> str:
    """
    Takes all subtask outputs and combines them into a coherent final report.

    WHY A THIRD LLM CALL INSTEAD OF JUST PRINTING THE RESULTS?
    Individual subtask outputs are like puzzle pieces — useful but separate.
    The synthesizer:
    - Finds connections between results
    - Eliminates redundancy
    - Creates a unified narrative with an executive summary
    - Adds strategic recommendations that emerge from the combined picture

    This is the most valuable stage — it's what transforms a list of answers
    into an actual business deliverable.
    """
    print(f"\n{'='*60}")
    print(f"STAGE 3: SYNTHESIZING FINAL REPORT")
    print(f"{'='*60}")

    # Build a structured block of all results to pass to the model
    combined_results = ""
    for subtask, result in zip(subtasks, results):
        combined_results += f"\n### Subtask {subtask['id']}: {subtask['title']}\n"
        combined_results += f"{result}\n"

    messages = [
        {
            "role": "system",
            "content": """You are a senior business consultant writing an executive report.
You have received the completed outputs of a multi-stage business analysis project.
Your job is to synthesize all the findings into one cohesive, professional final report.

The report MUST include:
1. Executive Summary (2-3 sentences capturing the core outcome)
2. Key Findings (one insight per subtask, drawn from the results)
3. Strategic Recommendations (3-5 actionable next steps)
4. Conclusion (brief, forward-looking)

Tone: professional, clear, confident. Avoid filler phrases like "It is important to note that..."
Do not just copy-paste the subtask outputs — synthesize, connect, and elevate them."""
        },
        {
            "role": "user",
            "content": (
                f"Original Business Goal:\n{goal}\n\n"
                f"Completed Subtask Outputs:\n{combined_results}\n\n"
                "Please write the final synthesized report."
            )
        }
    ]

    print("Generating final report...")
    return call_llm(messages, label="Synthesizer")


# =========================================================
# PIPELINE RUNNER
# =========================================================

def run_agent(goal: str) -> dict:
    """
    Orchestrates the full 3-stage pipeline for a given business goal.

    Returns a dict with:
    - goal:        the original input
    - subtasks:    the plan (list of dicts)
    - results:     individual subtask outputs (list of strings)
    - final_report: the synthesized final output (string)

    Returning everything in a dict makes it easy to:
    - Save to a file
    - Pass to another system
    - Display selectively in a UI
    """
    print(f"\n{'#'*60}")
    print(f"  GROQ ORCHESTRATION AGENT")
    print(f"{'#'*60}")

    # ── Stage 1: Plan ────────────────────────────────────────
    subtasks = plan_subtasks(goal)

    if not subtasks:
        print("[FATAL] Could not generate subtasks. Aborting.")
        return {}

    # ── Stage 2: Execute each subtask ────────────────────────
    results = []
    total = len(subtasks)

    for subtask in subtasks:
        result = execute_subtask(
            subtask=subtask,
            goal=goal,
            task_index=subtask["id"],
            total_tasks=total
        )
        results.append(result)

    # ── Stage 3: Synthesize ───────────────────────────────────
    final_report = synthesize_results(goal, subtasks, results)

    return {
        "goal":         goal,
        "subtasks":     subtasks,
        "results":      results,
        "final_report": final_report
    }


# =========================================================
# OUTPUT FORMATTER
# =========================================================

def print_final_output(agent_output: dict):
    """
    Prints the complete pipeline output in a readable format.
    In a real app you'd save this to a file, database, or UI instead.
    """
    if not agent_output:
        return

    print(f"\n\n{'#'*60}")
    print(f"  FINAL REPORT")
    print(f"{'#'*60}")
    print(f"\nOriginal Goal:\n  {agent_output['goal']}\n")

    print(f"Plan Executed ({len(agent_output['subtasks'])} subtasks):")
    for task in agent_output["subtasks"]:
        print(f"  ✓ [{task['id']}] {task['title']}")

    print(f"\n{'─'*60}")
    print("SYNTHESIZED FINAL REPORT:")
    print(f"{'─'*60}\n")
    print(agent_output["final_report"])
    print(f"\n{'#'*60}\n")

    # Optionally save to a text file
    save = input("Save report to file? (y/n): ").strip().lower()
    if save == "y":
        filename = "business_report.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"BUSINESS GOAL:\n{agent_output['goal']}\n\n")
            f.write("=" * 60 + "\n\n")

            f.write("SUBTASK BREAKDOWN:\n")
            for task, result in zip(agent_output["subtasks"], agent_output["results"]):
                f.write(f"\n[{task['id']}] {task['title']}\n")
                f.write(f"Task: {task['description']}\n")
                f.write(f"Output:\n{result}\n")
                f.write("-" * 40 + "\n")

            f.write("\n" + "=" * 60 + "\n\n")
            f.write("FINAL SYNTHESIZED REPORT:\n\n")
            f.write(agent_output["final_report"])
        print(f"  Saved to: {filename}")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    print("GROQ Multi-Stage Business Orchestration Agent")
    print("=" * 60)
    print("Enter a complex business goal and watch the agent:")
    print("  1. Break it into structured subtasks (JSON)")
    print("  2. Execute each subtask independently")
    print("  3. Synthesize everything into a final report")
    print("=" * 60)

    # Example prompts to try:
    # "Launch a new SaaS product for freelancers in North Africa"
    # "Build a content marketing strategy for a local coffee brand"
    # "Expand our e-commerce store into the European market"

    print("\nExample goal: 'Launch a SaaS product for freelancers in Algeria'")
    print("Or type your own.\n")

    goal = input("Your business goal: ").strip()

    if not goal:
        # Default demo goal if user just pressed Enter
        goal = "Launch a SaaS invoicing product targeting freelancers in Algeria"
        print(f"Using demo goal: {goal}")

    output = run_agent(goal)
    print_final_output(output)