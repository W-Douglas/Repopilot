# Repopilot
<p align="center">
  <img src="./images/Overview.png" alt="Overview" width="80%">
</p>
## Overview
RepoPilot is a lightweight Coding Agent system designed for Python code repositories, aimed at transforming code-fixing tasks from simply ‘having a large language model generate code’ into a controlled, verifiable and traceable engineering workflow.

The system takes as input the path to a local code repository and user-provided issue or bug descriptions, or pytest error logs. RepoPilot automatically handles understanding the repository structure, retrieving relevant code, planning a fix, generating structured editing instructions, making controlled code modifications, verifying with pytest tests, retrying in the event of failure, and generating a final report.

## Getting Started
### 1.Clone the Repository
```bash
git clone https://github.com/W-Douglas/Repopilot.git
cd Repopilot
```
### 2. Create a Python Environment
```bash
conda create -n repopilot python=3.10
conda activate repopilot
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4.Project Structure
```bash
Repopilot/
├── repomap/       # Build repository-level structural maps
├── retriever/     # Retrieve relevant code context
├── planner/       # Generate repair plans
├── coder/         # Generate structured code edit instructions
├── tools/         # Controlled file/search/git/test tools
└── README.md
```
