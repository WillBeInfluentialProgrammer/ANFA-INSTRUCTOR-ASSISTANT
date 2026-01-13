import os
import json
import re
from typing import Dict, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()


class QuestionGenerator:
    """
    Universal JSON-based question generator for ANF / Legal documents
    Supports:
      - Manuals   → sections / section
      - Laws      → rules / rule
    """

    def __init__(self):
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.01,
            max_tokens=2048,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

    # --------------------------------------------------
    # STRUCTURE DETECTION & NORMALIZATION
    # --------------------------------------------------

    def _normalize(self, data: Dict) -> Tuple[Dict[str, Dict], str]:
        if "sections" in data:
            array_key, id_key, label = "sections", "section", "Section"
        elif "rules" in data:
            array_key, id_key, label = "rules", "rule", "Rule"
        else:
            raise ValueError("Unsupported JSON structure")

        normalized = {
            str(item[id_key]): {
                "title": item.get("title", ""),
                "body": item.get("body", ""),
            }
            for item in data[array_key]
            if id_key in item
        }

        if not normalized:
            raise ValueError("No valid content found")

        return normalized, label

    def _context_json(self, normalized: Dict, label: str) -> str:
        return json.dumps(
            [
                {"id": k, "title": v["title"], "body": v["body"]}
                for k, v in normalized.items()
            ],
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    def _validate(self, text: str, valid_ids: set, label: str):
        refs = re.findall(fr"Reference:\s*{label}\s*([A-Za-z0-9\-]+)", text)
        invalid = [r for r in refs if r not in valid_ids]
        if invalid:
            print(f"⚠️ Warning: Invalid {label} reference(s) found: {invalid}")
            print(f"✅ Valid {label}s available: {sorted(valid_ids)}")

    # --------------------------------------------------
    # PROMPT BUILDER
    # --------------------------------------------------

    def _prompt(self, task: str, context: str, label: str, valid_ids: str) -> PromptTemplate:
        # Escape curly braces in context to prevent LangChain from treating them as variables
        escaped_context = context.replace("{", "{{").replace("}", "}}")
        
        template = f"""
You are a legal instructor.

The JSON below is the ONLY source of truth.
Use ONLY the provided IDs: {valid_ids}

JSON CONTEXT:
{escaped_context}

STRICT RULES:
1. Do NOT invent references like "Appendix", "Introduction", "Chapter", etc.
2. ONLY use the IDs listed above: {valid_ids}
3. Reference format MUST be:
   Reference: {label} <ID>
4. Use exactly ONE reference per answer
5. Each reference MUST exist in the JSON above

{task}
"""
        return PromptTemplate(template=template, input_variables=[])

    # --------------------------------------------------
    # PUBLIC METHODS
    # --------------------------------------------------

    def generate_mcqs(self, data: Dict, n: int = 5) -> str:
        normalized, label = self._normalize(data)
        context = self._context_json(normalized, label)
        valid_ids = ", ".join(sorted(normalized.keys()))

        task = f"""
Create {n} MCQs.

FORMAT:
Q[N]. Question?
A) Option
B) Option (Correct)
C) Option
D) Option
Reference: {label} <ID>

REMEMBER: Only use IDs from: {valid_ids}
"""

        prompt = self._prompt(task, context, label, valid_ids)
        output = (prompt | self.llm).invoke({}).content.strip()
        self._validate(output, set(normalized.keys()), label)
        return output

    def generate_descriptive(self, data: Dict, n: int = 5) -> str:
        normalized, label = self._normalize(data)
        context = self._context_json(normalized, label)
        valid_ids = ", ".join(sorted(normalized.keys()))

        task = f"""
Create {n} descriptive questions.

FORMAT:
Q[N]. Question
Reference: {label} <ID>

REMEMBER: Only use IDs from: {valid_ids}
"""

        prompt = self._prompt(task, context, label, valid_ids)
        output = (prompt | self.llm).invoke({}).content.strip()
        self._validate(output, set(normalized.keys()), label)
        return output

    def answer_question(self, data: Dict, question: str) -> str:
        normalized, label = self._normalize(data)
        context = self._context_json(normalized, label)
        valid_ids = ", ".join(sorted(normalized.keys()))

        task = f"""
Question: {question}

INSTRUCTIONS:
1. Provide a COMPREHENSIVE and DETAILED answer from the JSON context
2. Include ALL relevant subsections, clauses, and provisions
3. If a section has multiple subsections (1), (2), (3), etc., include ALL of them
4. If subsections have clauses (a), (b), (c), etc., include ALL relevant clauses
5. Do NOT summarize - provide the COMPLETE content
6. Use proper formatting with bullet points or numbering for subsections
7. End with the section reference

FORMAT:
[Provide detailed answer with all subsections and clauses]

Reference: {label} <ID>

If not found, say exactly:
"I cannot find this information in the provided document."

REMEMBER: Be thorough and comprehensive, not brief.
"""

        prompt = self._prompt(task, context, label, valid_ids)
        output = (prompt | self.llm).invoke({}).content.strip()
        self._validate(output, set(normalized.keys()), label)
        return output

    def generate_scenario_assignment(self, data: Dict) -> str:
        """Generate comprehensive scenario-based assignment"""
        normalized, label = self._normalize(data)
        context = self._context_json(normalized, label)
        valid_ids = ", ".join(sorted(normalized.keys()))

        # Escape curly braces in context for template
        escaped_context = context.replace("{", "{{").replace("}", "}}")

        task = f"""
You are a legal instructor creating a realistic scenario-based assignment.

Create ONE realistic scenario-based assignment using the legal provisions in the context below.

CONTEXT (Your ONLY source of legal provisions):
{escaped_context}

⚠️ CRITICAL RULES - MUST FOLLOW:
1. Create a realistic narrative that MATCHES the law provided in context (e.g., if context is Money Laundering Act, create a money laundering scenario; if ANF Act, create narcotics scenario)
2. Reference ONLY laws/sections that exist in the context above: {valid_ids}
3. When citing sections, verify they exist in context
4. For penalties: ONLY use penalties explicitly stated in context sections
5. If context doesn't contain specific penalty amounts, state "penalties as prescribed in relevant laws"
6. Do NOT invent section numbers or reference laws not provided in context

REQUIRED FORMAT (Use this EXACT structure):

**Scenario Title**: "[Descriptive title matching the law in context]"

**Narrative**:
[Write 3-5 sentences describing a realistic situation that matches the legal framework in context. If context is about money laundering, write about money laundering activities. If about narcotics, write about narcotics offences. Make it engaging but ensure any legal actions mentioned can be supported by provisions in the context.]

**Structured Analysis**:

1. **The Crime (Legal Act)**
   - Which specific offence(s) have been committed?
     [Describe the offence based on the law in context]
   - Under which section(s)/article(s)?
     [Cite ONLY sections that exist in the context: {valid_ids}]

2. **The Criminality (Tendency / Risk Factors)**
   - What circumstances/factors contributed to this criminal behavior?
     [Describe circumstances like organized crime connections, patterns of behavior]
   - What risk indicators were present?
     [Describe warning signs or indicators that led to investigation]

3. **The Criminal Psychology (Mindset / Motivation)**
   - What might have motivated the offender(s)?
     [Discuss likely motivations like financial gain, organized crime pressure]
   - What psychological factors are evident?
     [Discuss antisocial behavior, lack of empathy, risk-taking]

4. **Criminological Theory**
   - Which criminological theory/theories apply?
     [Apply theories like rational choice theory, social learning theory, strain theory]
   - How do they explain this behavior?
     [Explain how the theory accounts for the criminal behavior]

5. **Professional Response / Intervention**
   - What legal actions should law enforcement take?
     [Based on powers in context - cite specific sections from: {valid_ids}]
   - What procedures must be followed?
     [Reference specific sections from: {valid_ids}]
   - What penalties apply?
     [IMPORTANT: ONLY cite penalties explicitly mentioned in the context sections]

VERIFICATION: Available sections are {valid_ids}

Generate the scenario now:
"""

        # Create prompt without calling _prompt method (direct template)
        template = PromptTemplate(template=task, input_variables=[])
        output = (template | self.llm).invoke({}).content.strip()
        return output
