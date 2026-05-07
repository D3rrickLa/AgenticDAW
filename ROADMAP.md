This project is a demostration on using Agentic AI in an 'real-world' application. The concept is that we have a DAW (Digital Audio Workstation) as we are producing a song/mixing audio, the "AI" agent has an overview on what we are doing and based on some commands/instructions/.txt file, will make suggestions on what to fix/add. Examples could be 'grouping' your tracks, making a 'send/bus', fixing BPM, tempo issues. This isn't a full DAW, but a 'pseudo-abstraction' of one.


Tech stack/things to consider:
- Next.js
- Python/FastAPI
- Python - we will be using Ollama for this, a light model (3b or 7b)
- How to implement an Agentic system


## file structure

- main.py 
- agent
    - loop
    - prompts
    - memory
    - parser
    - state

- tools
    - calculator
    - filesystem
    - DAW
    - audio
    - registry

- models
    - ollama 

- data
    - sessions

- tests


# DAW Integration Options

## Option A - REAPER

### Advantages

* Strong scripting API
* Python/Lua support
* OSC support
* Lightweight
* Automation-friendly

### Potential Features

* Track creation
* FX insertion
* Rendering
* Marker editing
* Session automation

---

## Option B - Browser-Based AI DAW

### Stack

* React
* Tone.js
* Web Audio API

### Advantages

* Full control
* Structured state
* Easier AI integration
* Cross-platform
* Easier deployment

### Recommended for MVP

This is currently the preferred prototype direction.

---

# MVP Scope

## Initial Demo Goals

User should be able to:

* Chat with the agent
* Create tracks
* Set BPM
* Add effects
* Analyze audio
* Render previews
* Save/load sessions

---

# Proposed MVP Flow

User:

> "Create a chill synthwave loop"

Agent:

1. Creates project
2. Sets BPM
3. Adds drum track
4. Adds synth track
5. Applies reverb
6. Generates arrangement
7. Renders preview

---

# Development Roadmap

# Phase 1 - Foundation

## Goals

* Modular architecture
* Tool registry
* Stable agent loop
* Structured state
* Basic filesystem tools

## Deliverables

* Refactored codebase
* Generic Tool class
* Improved prompting
* Session persistence

---

# Phase 2 - Project Engine

## Goals

* JSON project state
* Track system
* Timeline representation
* Effect chains

## Deliverables

* create_track tool
* set_volume tool
* BPM control
* Save/load projects

---

# Phase 3 - Audio Intelligence

## Goals

* Audio-aware reasoning
* Basic DSP analysis
* Waveform processing

## Deliverables

* BPM detection
* Key detection
* Loudness analysis
* Silence trimming

---

# Phase 4 - Agent Planning

## Goals

* Multi-step planning
* Verification
* Task tracking
* Reflection improvement

## Deliverables

* Planner module
* Executor module
* State tracking
* Retry logic

---

# Phase 5 - Frontend

## Goals

* Interactive UI
* Session visualization
* Track editor
* Chat interface

## Deliverables

* React frontend
* Timeline UI
* Track inspector
* Reasoning panel

---

# Phase 6 - Advanced Features

## Potential Features

* Multi-agent workflows
* Plugin recommendation
* Auto-mixing suggestions
* Arrangement generation
* Voice control
* Real-time assistant mode

---

# Safety Considerations

## Important Rules

* Never execute arbitrary shell commands blindly
* Restrict filesystem access
* Validate all tool arguments
* Require confirmations for destructive actions
* Limit infinite reasoning loops

## Example

```python
SAFE_DIRS = [
    "/projects/music"
]
```

---

# Reliability Challenges

Agent systems often fail because of:

* Hallucinated tool usage
* Infinite loops
* Invalid state transitions
* Poor planning
* Context overflow

## Mitigation Strategies

* Validation layers
* Step limits
* Retry policies
* Structured state
* Tool result verification
* Explicit planner/executor separation

---

# Design Philosophy

The project should remain:

* Local-first
* Inspectable
* Modular
* Deterministic where possible
* Tool-driven
* Audio-aware
* Extensible

The system should avoid unnecessary complexity in early stages.

---

# Long-Term Vision

Possible future direction:

> An AI-native music production environment where agents collaborate with humans through structured project state, audio intelligence, and autonomous workflows.

Potential applications:

* Music production
* Podcast editing
* Sound design
* Film/game audio
* Live performance systems
* Generative composition

---

# Current Status

## Completed

* Basic tool-calling agent
* Reflection loop
* Multi-step execution
* Ollama integration
* Safe calculator tool example
* Tenant demo environment

## Next Immediate Steps

1. Refactor architecture
2. Implement Tool abstraction
3. Add filesystem tools
4. Create structured project state
5. Build planner/executor separation
6. Prototype browser-based DAW environment

---

# Guiding Principle

The project is not simply:

> “AI inside a DAW.”

It is closer to:

> “A programmable music operating system with an AI orchestration layer.”
