# Agentic DAW

## Vision

AI Agent DAW Helper is an experimental autonomous music-production assistant that combines:

* Local LLM reasoning
* Tool calling
* Audio analysis
* DAW/project-state manipulation
* Workflow automation
* Conversational interaction

The goal is not to replace music producers, but to create an intelligent production copilot capable of assisting with:

* Project organization
* Track setup
* Arrangement
* Mixing preparation
* Audio analysis
* Rendering workflows
* Session memory
* Creative iteration

The system is designed to run locally-first using Ollama and lightweight tooling.

---

# Core Concept

The project is built around an "agent + tools" architecture.

Instead of hardcoding workflows, the LLM reasons about goals and decides which tools to use.

Example:

User:

> "Prepare the vocals for mixing and export a preview."

Agent:

1. Analyze vocal stems
2. Detect silence/noise
3. Normalize gain
4. Add cleanup chain
5. Render preview
6. Summarize changes

The intelligence comes from:

* Planning
* Tool orchestration
* Reflection loops
* Persistent memory

---

# Project Goals

## Primary Goals

* Build a local autonomous music-production assistant
* Create an extensible agent architecture
* Support multiple DAW/workflow integrations
* Enable structured project-state editing
* Add audio-aware reasoning capabilities
* Create a strong technical demo/MVP

## Secondary Goals

* Explore autonomous creative workflows
* Experiment with multi-agent music systems
* Investigate AI-native DAW interaction models
* Prototype a browser-based AI production environment

---

# High-Level Architecture

```text
User
  ↓
Frontend UI
  ↓
Agent Runtime
  ↓
Planner / Executor
  ↓
Tool System
  ↓
Audio Engine / Project State / Filesystem
```

---

# Recommended Stack

## LLM Layer

### Primary

* Ollama
* Qwen models
* DeepSeek models
* Llama-based models

### Potential Models

* qwen3.5:9b
* deepseek-r1
* llama3

---

# Backend

## Language

* Python

## Responsibilities

* Agent runtime
* Tool execution
* Memory management
* Planning
* Audio analysis
* Session management
* API server

---

# Frontend

## Recommended

* React
* TypeScript
* Tailwind

## Audio/UI

* Tone.js
* Web Audio API

## UI Components

* Chat interface
* Timeline/tracks
* Tool execution log
* Reasoning viewer
* Session memory panel
* Project inspector

---

# Suggested Repository Structure

```text
project/
│
├── main.py
├── agent/
│   ├── loop.py
│   ├── planner.py
│   ├── executor.py
│   ├── memory.py
│   ├── parser.py
│   ├── prompts.py
│   └── state.py
│
├── tools/
│   ├── registry.py
│   ├── filesystem/
│   ├── audio/
│   ├── daw/
│   ├── project/
│   └── utility/
│
├── models/
│   └── ollama_client.py
│
├── frontend/
│
├── sessions/
│
├── data/
│
└── tests/
```

---

# Agent Architecture

## Current Architecture

Current implementation uses:

* Tool calling
* Reflection loops
* Multi-step reasoning
* Conversation memory
* Tool execution feedback

## Future Architecture

```text
Goal
  ↓
Planner
  ↓
Task List
  ↓
Executor
  ↓
Tool Calls
  ↓
Verification
  ↓
Final Output
```

---

# Tool System

## Philosophy

Tools are the capability layer of the system.

The model should never directly manipulate the environment.
Instead, it uses validated tools.

---

# Tool Categories

## 1. Filesystem Tools

### Examples

* read_file
* write_file
* list_directory
* search_files
* rename_file

### Purpose

* Project organization
* Session persistence
* Asset management

---

## 2. Project-State Tools

These manipulate structured DAW/project state.

### Examples

* create_track
* delete_track
* set_volume
* add_effect
* set_bpm
* move_clip
* create_marker

### Purpose

* Deterministic DAW interaction
* Safer than UI automation
* Easier debugging

---

## 3. Audio Analysis Tools

### Examples

* detect_bpm
* detect_key
* analyze_loudness
* trim_silence
* detect_clipping
* transcribe_audio

### Libraries

* librosa
* torchaudio
* essentia

### Purpose

Give the agent audio-awareness.

---

## 4. Rendering Tools

### Examples

* render_preview
* export_stems
* bounce_track
* export_mixdown

---

## 5. Memory Tools

### Examples

* save_project_note
* retrieve_preferences
* remember_plugin_chain
* summarize_session

---

# Tool Abstraction Design

Recommended abstraction:

```python
class Tool:
    name: str
    description: str
    schema: dict

    def run(self, **kwargs):
        pass
```

Advantages:

* Dynamic registration
* Validation
* Easier scaling
* Tool introspection
* Auto-generated schemas

---

# Memory System

## Short-Term Memory

Conversation and active-task context.

## Long-Term Memory

Persistent user/project memory.

### Example

```json
{
  "project": "album_x",
  "favorite_synth": "Serum",
  "preferred_vocal_chain": ["EQ", "Compressor"]
}
```

## Possible Storage

### Early Stage

* JSON
* SQLite

### Later Stage

* Vector database
* Semantic retrieval

---

# Planning System

## Purpose

Improve long-task reliability.

## Example

Goal:

> "Prepare drums for mixing"

Generated plan:

1. Analyze drum stems
2. Normalize gain
3. Detect clipping
4. Add routing
5. Export preview

---

# Project-State Architecture

Instead of directly controlling a traditional DAW UI, the system should primarily manipulate a structured project state.

Example:

```json
{
  "bpm": 128,
  "tracks": [
    {
      "name": "drums",
      "volume": -4,
      "effects": ["compressor"]
    }
  ]
}
```

Advantages:

* Deterministic
* Easier debugging
* Easier syncing
* Browser-native
* Better AI integration

---