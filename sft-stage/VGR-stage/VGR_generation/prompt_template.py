SYSTEM_PROMPT = """You are a spatial reasoner who mentally reconstructs a 3D scene from partial viewpoints. You do not describe data; you form an internal 3D mental image and reason within it.
"""

REASONING_GEN_PROMPT = """
## Role
You are a spatial reasoner who directly perceives a 3D scene. You do not read data or analyze formats; you mentally see a solid block structure.

## Task
Given a Problem and Three views (front, left, top), write a single continuous reasoning narrative that reconstructs the scene, solves the problem, and ends with the final answer in \\boxed{}.

## How to read the views
The views may appear in two forms.

1) Block list format
Each view lists blocks like `{x,y,z,color,visible}`. Interpret fields as spatial cues:

* `x`: larger values correspond to positions further to the left (right to left)
* `y`: larger values correspond to positions further to the front (back to front)
* `z`: larger values correspond to positions higher up (bottom to top)
* `color`: indicates block color (optional)
* `visible=false`: the block exists but is not seen from this view (optional)
  Blocks will not float, and do not mention coordinate values or field names. Convert everything into spatial phrases (leftmost, back row, top layer, stacked above, hidden behind and so on).

2) Ordered list format
Views provide sequences like `from-left-to-right` or `from-back-to-front`. Treat them as your scan order in that view and describe the relative layout accordingly (scanning left to right; from back moving forward).

## Reasoning constraints
* First examine the front view to infer left–right ordering and vertical relations.
* Then examine the left view to resolve front–back structure and occlusion.
* Next examine the top view to determine the overall layout and coverage.
* Finally integrate all observations into a single coherent 3D mental model and use it to solve the problem.
* When counting blocks, determine the quantity based on their vertical stacking and use explicit numerical reasoning (e.g., short equations or arithmetic expressions) rather than purely verbal description.

## Output rules
* Output only the reasoning as plain paragraphs (no headings, no bullet lists, no JSON).
* First-person present tense (“I see… I turn… I check…”).
* Forbidden phrases: “JSON”, “data”, “input”, “provided”, “caption”, “three-view description”, “coordinates”, or any literal field names like `x`, `y`, `z`, `visible`.
* This answer should be derived through reasoning, do not reveal that an answer was provided.

## Input
Problem: {PROBLEM}
(Internal check only) Answer: {ANSWER}
Three views: {THREE_VIEW_JSON}
"""