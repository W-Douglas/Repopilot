# Repopilot
## Overview
RepoPilot is a lightweight Coding Agent system designed for Python code repositories, aimed at transforming code-fixing tasks from simply ‘having a large language model generate code’ into a controlled, verifiable and traceable engineering workflow.

The system takes as input the path to a local code repository and user-provided issue or bug descriptions, or pytest error logs. RepoPilot automatically handles understanding the repository structure, retrieving relevant code, planning a fix, generating structured editing instructions, making controlled code modifications, verifying with pytest tests, retrying in the event of failure, and generating a final report.

