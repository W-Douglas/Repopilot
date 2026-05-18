"""
LLM Agent Node - LangGraph node that uses LLM with tool calling.

This module provides LangGraph nodes that:
1. Use LLM to decide which tool to call
2. Execute the tool via ToolManager
3. Return results to LLM for next iteration

The agent loop: LLM -> Tool Call -> Execute -> LLM -> ...
"""

from typing import Dict, Any, List, Optional, Callable, Literal
from dataclasses import dataclass
import json

# Try relative imports first, then absolute
try:
    from ..llm import BaseLLM, ChatMessage, LLMFactory
    from ..graph.state import RepoPilotState, TaskStatus
except ImportError:
    try:
        from llm import BaseLLM, ChatMessage, LLMFactory
        from graph.state import RepoPilotState, TaskStatus
    except ImportError:
        # For workflow integration, use absolute imports
        from repomap.llm import BaseLLM, ChatMessage, LLMFactory
        from repomap.graph.state import RepoPilotState, TaskStatus

from .llm_tool_adapter import AgentToolSystem, FunctionCall


@dataclass
class AgentConfig:
    """Configuration for an LLM agent node."""
    name: str
    description: str
    tools: List[str]  # Tools this agent can use
    system_prompt: str
    max_iterations: int = 5
    stop_on_error: bool = True


class LLMAgentNode:
    """
    A LangGraph node that runs an LLM with tool calling.

    Usage:
        agent_node = LLMAgentNode(
            config=AgentConfig(
                name="PlannerAgent",
                tools=["read_file", "search_code"],
                system_prompt="You are a code planner..."
            )
        )

        # In workflow:
        workflow.add_node("planner", agent_node.execute)
    """

    def __init__(
        self,
        config: AgentConfig,
        llm: Optional[BaseLLM] = None,
        tool_system: Optional[AgentToolSystem] = None,
    ):
        self.config = config
        self.llm = llm
        self.tool_system = tool_system

    def execute(self, state: RepoPilotState) -> RepoPilotState:
        """
        Execute the agent for one turn.

        This performs one iteration of: LLM -> Tool -> Result -> LLM

        Args:
            state: Current workflow state

        Returns:
            Updated state with agent results
        """
        # Get LLM from state or use configured
        llm = self._get_llm(state)
        if llm is None:
            state["error"] = f"No LLM configured for {self.config.name}"
            state["status"] = TaskStatus.FAILED.value
            return state

        # Get tool system
        tool_system = self._get_tool_system(state)

        # Build messages
        messages = self._build_messages(state)

        # Get available tools
        tool_schemas = tool_system.adapter.get_function_schemas()

        # Call LLM with tools
        try:
            response = self._call_llm_with_tools(llm, messages, tool_schemas)

            if response.content:
                # Store LLM response
                state[f"{self.config.name}_response"] = response.content

                # Parse and execute tool calls
                tool_calls = tool_system.adapter.parse_tool_calls(response.content)

                if tool_calls:
                    state[f"{self.config.name}_tool_calls"] = [
                        {"name": tc.name, "args": tc.arguments}
                        for tc in tool_calls
                    ]

                    # Execute first tool call (for simple agents)
                    # For complex agents, could execute all and collect results
                    for tc in tool_calls:
                        result = tool_system.execute_function_call(tc, self.config.name)

                        # Store result in state
                        state[f"tool_result_{tc.name}"] = result.to_dict()
                        state["tool_results"] = state.get("tool_results", []) + [result.to_dict()]

                        if not result.success and self.config.stop_on_error:
                            state["error"] = result.error
                            break

                    state["status"] = f"{self.config.name}_tool_called"
                else:
                    # No tool call, LLM provided direct response
                    state[f"{self.config.name}_final"] = response.content
                    state["status"] = f"{self.config.name}_completed"

        except Exception as e:
            state["error"] = f"{self.config.name} failed: {str(e)}"
            state["status"] = TaskStatus.FAILED.value

        return state

    def _get_llm(self, state: RepoPilotState) -> Optional[BaseLLM]:
        """Get LLM from state or config."""
        if self.llm:
            return self.llm

        # Try from state
        llm_config = state.get("llm_config", {})
        api_key = llm_config.get("api_key")

        if not api_key:
            return None

        try:
            model = llm_config.get("model", "deepseek-chat")
            provider = llm_config.get("provider", "deepseek")
            return LLMFactory.create(model=model, provider=provider, api_key=api_key)
        except Exception:
            return None

    def _get_tool_system(self, state: RepoPilotState) -> AgentToolSystem:
        """Get or create tool system."""
        if self.tool_system:
            return self.tool_system

        repo_path = state.get("repo_path", ".")
        return AgentToolSystem(root=repo_path)

    def _build_messages(self, state: RepoPilotState) -> List[ChatMessage]:
        """Build messages for the agent."""
        messages = []

        # System prompt
        messages.append(ChatMessage(
            role="system",
            content=self.config.system_prompt
        ))

        # Add context from state
        context = []
        if "issue" in state:
            context.append(f"Issue: {state['issue']}")
        if "retrieved_context" in state:
            context.append(f"Context: {len(state['retrieved_context'])} chunks available")
        if "bug_analysis" in state:
            context.append(f"Analysis: {state['bug_analysis'][:200]}...")

        if context:
            messages.append(ChatMessage(
                role="system",
                content=f"Context:\n" + "\n".join(context)
            ))

        # Add tool results from previous calls
        if "tool_results" in state:
            for tr in state["tool_results"]:
                messages.append(ChatMessage(
                    role="user",
                    content=f"Tool result: {tr['output'][:500]}"
                ))

        # Task prompt
        task = state.get("issue", state.get("task", ""))
        messages.append(ChatMessage(
            role="user",
            content=f"Task: {task}\n\nAvailable tools: {', '.join(self.config.tools)}\n\nUse a tool if needed, otherwise respond directly."
        ))

        return messages

    def _call_llm_with_tools(
        self,
        llm: BaseLLM,
        messages: List[ChatMessage],
        tool_schemas: List[Dict[str, Any]]
    ) -> Any:
        """Call LLM with tool schemas."""
        # Prepare messages for LLM
        llm_messages = [m.to_dict() for m in messages]

        # For OpenAI-style models with function calling
        if hasattr(llm, 'chat_with_tools'):
            return llm.chat_with_tools(llm_messages, tool_schemas)

        # For models without native function calling
        # Include tool descriptions in prompt
        tools_prompt = self._build_tools_prompt(tool_schemas)
        llm_messages[0] = {
            "role": "system",
            "content": messages[0].content + "\n\n" + tools_prompt
        }

        response = llm.chat(
            [ChatMessage.from_dict(m) for m in llm_messages]
        )

        return type('Response', (), {'content': response.content})()


    def _build_tools_prompt(self, schemas: List[Dict[str, Any]]) -> str:
        """Build prompt section describing available tools."""
        lines = ["\nAvailable functions (use in JSON format):"]

        for schema in schemas:
            name = schema.get("name", "")
            desc = schema.get("description", "")
            params = schema.get("parameters", {})

            props = params.get("properties", {})
            required = params.get("required", [])

            lines.append(f"\n{name}:")
            lines.append(f"  Description: {desc}")

            if props:
                lines.append("  Parameters:")
                for pname, pdef in props.items():
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    required_mark = "(required)" if pname in required else "(optional)"
                    lines.append(f"    - {pname}: {ptype} {required_mark} - {pdesc}")

        return "\n".join(lines)


def create_planner_agent(llm: Optional[BaseLLM] = None) -> LLMAgentNode:
    """Create a planner agent node."""
    config = AgentConfig(
        name="PlannerAgent",
        description="Analyzes issues and generates fix plans",
        tools=["read_file", "get_repo_tree", "search_code"],
        system_prompt="""You are an expert code planner. Your job is to:
1. Analyze the issue and code context
2. Identify which files need to be modified
3. Determine the fix strategy

You can use tools to read files and search for code patterns.
When you have enough information, provide a detailed fix plan.
"""
    )
    return LLMAgentNode(config=config, llm=llm)


def create_coder_agent(llm: Optional[BaseLLM] = None) -> LLMAgentNode:
    """Create a coder agent node."""
    config = AgentConfig(
        name="CoderAgent",
        description="Generates code edits based on fix plans",
        tools=["read_file", "write_file", "search_code"],
        system_prompt="""You are an expert code editor. Your job is to:
1. Read the target files
2. Generate precise edit instructions
3. Use write_file to apply changes

Use minimal edits - prefer targeted replacements over wholesale rewrites.
Always include clear old_text and new_text for replacements.
"""
    )
    return LLMAgentNode(config=config, llm=llm)


def create_reviewer_agent(llm: Optional[BaseLLM] = None) -> LLMAgentNode:
    """Create a reviewer agent node."""
    config = AgentConfig(
        name="ReviewAgent",
        description="Reviews changes and runs tests",
        tools=["read_file", "run_pytest", "get_git_diff"],
        system_prompt="""You are an expert code reviewer. Your job is to:
1. Review the changes made
2. Run tests to verify correctness
3. Identify any issues or improvements

Use run_pytest to verify changes work correctly.
"""
    )
    return LLMAgentNode(config=config, llm=llm)