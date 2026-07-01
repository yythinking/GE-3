# agent/prompts.py

# ==============================================================================
# 策略核心：为了提升 EE (有效信息覆盖率)，我们必须让 LLM 变得“贪婪”。
# 它不应该问已知的，而应该盯着“未知”的、“提了一嘴但没展开”的实体猛攻。
# ==============================================================================

# --- 1. 悬念追问 (The Cliffhanger Strategy) ---
# 目标：基于当前文本，挖掘那些“被提及但未解释”的概念，强制 RAG 检索新 Chunk。
# 变更：从 "expand on this" 改为 "target unexplained terms".
# GENERATE_EVIDENCE_BASED_QUESTIONS_PROMPT = """
# You are a meticulous investigator analyzing a fragment of a database.
# Your goal is to find "Cliffhangers" — specific terms, names, or events mentioned in the text but NOT fully explained.

# Input Text:
# "{corpus}"

# Task: Generate 3-5 specific questions to retrieve missing details.

# Rules:
# 1. **Target the Unknown**: If the text says "Project X was cancelled", do NOT ask "Was Project X cancelled?". Ask "Why was Project X cancelled?" or "Who cancelled Project X?".
# 2. **Follow Proper Nouns**: Prioritize questions about capitalized names, places, or specific IDs found in the text.
# 3. **Bridge Concepts**: If two concepts appear together (e.g., "Drug A interacts with Protein B"), ask specifically about that interaction.
# 4. **Short & Specific**: Keep questions under 15 words.

# Output (One question per line):
# """
# ==============================================================================
# 混合策略 (Hybrid Strategy): 平衡 SS 和 EE
# 我们不再只问“悬念”，而是强制模型生成两类问题：
# 1. 锚定问题 (Anchoring): 针对已知事实提问 -> 保证高 SS (Baseline Aligned)
# 2. 扩展问题 (Expansion): 针对未知概念提问 -> 保证高 EE (Novelty)
# ==============================================================================

GENERATE_EVIDENCE_BASED_QUESTIONS_PROMPT = """
You are a strategic investigator exploring a database.
Your task is to generate 4 questions based on the provided text:
- 2 Questions must be "Verification Questions" (Checking facts explicitly stated).
- 2 Questions must be "Discovery Questions" (Asking about mentioned but unexplained terms).

Input Text:
"{corpus}"

Rules:
1. **Verification (High Confidence)**: Ask about specific numbers, definitions, or outcomes explicitly written in the text. (e.g., "What is the specific dosage of Drug X?").
2. **Discovery (High Novelty)**: Identify a proper noun or term that is mentioned but NOT defined, and ask about it. (e.g., "What are the side effects of Drug X?" if side effects are not listed).
3. **Format**: Output one question per line.

Output (4 Questions):
"""


# --- 2. 广义锚点提取 (Domain-Agnostic Anchor Extraction) ---
# 目标：建立候选词池。
# 变更：彻底移除 "medical" 等限定词，改为提取所有 "专有名词" 和 "技术术语"。
# 这直接解决了 HarryPotter 数据集提取不到东西的问题。
EXTRACT_ANCHORS_PROMPT = """
Analyze the text below and extract potential "Search Keys" for a database query.
We need entities that are likely to have their own dedicated definitions or sections in the database.

Input Text:
"{corpus}"

Extraction Rules:
1. **Proper Nouns (High Priority)**: Extract all capitalized names (People, Places, Organizations, Products).
   - e.g., "Harry Potter", "Hogwarts", "Aspirin", "Python".
2. **Technical Terms**: Extract specific jargon or scientific terms.
   - e.g., "Hypertension", "Gradient Descent", "Expelliarmus".
3. **Events/Acts**: Extract specific named events.
   - e.g., "The Battle of Hogwarts", "Tax Reform Act".
4. **Exclude**: Generic stop words (e.g., "System", "Help", "Document", "User", "Question").

Format: Output one entity per line.

Output (Entities only):
"""

# --- 3. 跨步跳跃 (The Pivot) ---
# 目标：当对话停滞时，利用历史实体进行大幅度的语义跳跃。
# 变更：增加了“关系”和“起源”类问题，避免只问定义（定义往往还在原来的 Chunk 里）。
GENERATE_PIVOT_QUESTION_PROMPT = """
Context: We are exploring a knowledge base and need to switch topics.
Target Entity: "{anchor}"

Task: Write a specific question about "{anchor}" that forces the system to look up new information.
Try to ask about:
- Relationships: "Who is connected to {anchor}?"
- Causes/Effects: "What are the side effects of {anchor}?" or "What triggers {anchor}?"
- Origins: "Where did {anchor} come from?"

Question:
"""

# --- 4. 拒绝检测 (Rejection Check) ---
# 保持不变，这部分逻辑是正确的。
CONFIRM_ANSWER_REJECTED_PROMPT = """
Analyze the following response from an AI assistant.
Determine if it is a refusal/rejection to answer (e.g., "I don't know", "I cannot help", "Not mentioned in context").

Response:
"{answer}"

Does this response contain valid information? 
Answer "YES" if it has valid info (even if partial).
Answer "NO" if it is a refusal, empty, or purely conversational filler without facts.
Only output YES or NO.
"""

# --- 5. 答案清洗 (Soft Cleaner) ---
# 保持你的修改，防止过度清洗导致信息丢失。
GET_ANSWER_ONLY_PROMPT = """
You are a text cleaner. Your job is to extract the signal from the noise.

Input Text:
"{answer}"

Instructions:
1. Remove conversational fillers (e.g., "I hope this helps", "System:", "Based on the provided context").
2. **CRITICAL**: If the text contains ANY factual keywords, list items, or definitions, KEEP THEM. 
3. Do not summarize; try to preserve the original phrasing of facts.
4. Avoid using additional explanations (e.g., “After analyzing the input text,” “Here is the cleaned text”), and focus on the facts.

Cleaned Text:
"""