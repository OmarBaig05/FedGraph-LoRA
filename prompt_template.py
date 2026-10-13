from typing import Literal, List, Annotated
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class ClassificationOutput(BaseModel):
    category: Literal["query", "preference", "both"] = Field(
        ..., description="The type of input: query, preference, or both."
    )
    genre_filter: List[str] = Field(
        default_factory=list,
        description="List of broad topics like food, comedy, movie, books, etc., only for 'query'. Empty otherwise."
    )
    preference_type_filter: List[Literal["like", "dislike", "prefer", "avoid", "neutral"]] = Field(
        default_factory=list,
        description="List of user preference types, like prefer, avoid, etc. Only for 'query'. Empty otherwise."
    )

class FeatureObject(BaseModel):
    """Represents a single implicit feature extracted from conversation."""
    Standard_fine_tuning_prompt: str = Field(
        ..., description="A system prompt followed by user and assistant messages that demonstrate the implicit feature"
    )

class FeaturesOutput(BaseModel):
    """Output parser for implicit features extraction."""
    features: Annotated[List[str], Field(default_factory=list)]

class JudgeResult(BaseModel):
    verdict: int  # 1 if correct, 0 if wrong
    feedback: str  # short feedback


judge_parser = PydanticOutputParser(pydantic_object=JudgeResult)
parser = PydanticOutputParser(pydantic_object=ClassificationOutput)

classification_prompt = PromptTemplate(
    template="""
You are an AI assistant that classifies user inputs. Your task is to:

1. Determine if the input is a:
   - "query": a question
   - "preference": a statement expressing a like/dislike/preference (explicit feature)
   - "both": question and preference both

2. If the input is a "query" or "both", also extract:
   - genre_filter: list of general categories relevant to the query (e.g., ["food", "movie", "comedy", "books"])
   - preference_type_filter: inferred preferences from the user, such as ["like", "prefer", "avoid", "dislike"]

Examples:

Input: "What are the best romantic movies to watch?"
Output: category="query", genre_filter=["movie", "romantic"], preference_type_filter=["prefer"]

Input: "I dislike horror books."
Output: category="preference", genre_filter=[], preference_type_filter=[]

Input: "Can you suggest some vegetarian recipes? I want to avoid meat."
Output: category="both", genre_filter=["food", "vegetarian"], preference_type_filter=["avoid"]

Examples of INCORRECT responses:
- "The input is a query about movies." (Not JSON)
- "Here's the classification: json" (Extra text around JSON)
- "```json\njson\n```" (Markdown formatting)


Now classify the following user input:

User Input: {user_input}

Respond in this format:
{format_instructions}
""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

extract_preferences_prompt = PromptTemplate(
    input_variables=["user_prompt"],
    template="""
You are a smart assistant helping to personalize user interactions.
Your job is to analyze the given prompt and determine if it contains:

1. Explicit preferences (clearly stated preferences like "I like", "I prefer", etc., one or two word genres like "movie", "book", "music", etc., and specific items like "comedy", "action", "romance", etc.)

If explicit or implicit preferences exist, return them in JSON with this format:
{{
  "explicit": [
    {{
      "type": "prefer",
      "genre": "<genre>",
      "specific item": "<specific item> (if applicable)",
      "conditions": {{
        "attribute": "value"
      }},
      "statement": "<Natural language statement describing this explicit preference, to the point explicit preference>"
    }}
  ],
}}

Make sure:
- Your JSON is valid and parseable.
- For explicit preferences, always include a "statement" summarizing them in natural language.
- Do NOT hallucinate conditions that are not implied.
- Only return the raw JSON object without any extra formatting like Markdown, explanation, or triple backticks.

Now analyze the following prompt and return the JSON:
---
Prompt: "{user_prompt}"
"""
)



strip_preferences_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="""
You are an assistant that processes user inputs. Your task is to extract or rewrite the input so that it contains **only the main question or query**, without any preference-related details.

- Remove all explicit or implicit preferences, such as likes, dislikes, desires, or constraints (e.g., "I like", "avoid", "prefer", "dislike", etc.).
- Keep only the **core question** the user is asking.
- Do NOT include any reasoning, explanation, or formatting — just return the cleaned question.

Examples:

Input: "Can you suggest some good Italian restaurants? I dislike crowded places."
Output: "Can you suggest some good Italian restaurants?"

Input: "What movies should I watch? I prefer comedy and avoid horror."
Output: "What movies should I watch?"

Now process the following input:

Input: {user_input}
Output:
"""
)


# ================================
# Implicit feature prompt template
# ================================

# Create parser for the features output
implicit_features_parser = PydanticOutputParser(pydantic_object=FeaturesOutput)

extract_implicit_features_prompt = PromptTemplate(
    input_variables=["chat_session"],
    template="""
You are an AI system that restructures chat sessions into fine-tuning training prompts.  

Each chat session includes:
- The user’s original input
- The assistant’s original response
- The user’s feedback on that response

Your task:
1. Keep the <|user|> section exactly as the original user’s query.
2. In the <|assistant|> section, rewrite the assistant’s response so that it satisfies the feedback.
   - Apply corrections, style changes, or detail improvements.
   - Do NOT explicitly mention the feedback in the response — just reflect it implicitly.
3. Return results in this exact JSON array format:

[{{
  "Standard_fine_tuning_prompt": "### Instruction:<the instructions present if any>### Input: <the query asked by the user>### Response: <the best possible response to that query>"
}},
{{
  "Standard_fine_tuning_prompt": "### Instruction:<the instructions present if any>### Input: <the query asked by the user>### Response: <the best possible response to that query>"
}}
]

Rules:
- If multiple user-feedback cycles exist, output multiple JSON objects in the array (one per query).
- Do not include the raw feedback text in the output.
- Do not include the original assistant response directly, only the corrected one.
- If no valid transformation can be done, output exactly: none
- No explanations, no comments, only the JSON array or "none".

Now process the following chat session:
{chat_session}
"""
)



judge_prompt = PromptTemplate.from_template(
    """
You are an evaluator LLM acting as a strict judge. 
You will be given:
- Query: {query}
- Ground Truth: {ground_truth}
- Preference: {preference}
- Response: {response}

Your evaluation rules:
1. Compare the Response with the Ground Truth and check if it aligns with the Preference.
2. If the Response is correct → return JSON with feedback "Correct." and verdict=1.
3. If the Response is wrong → return JSON with a short and clear feedback, then verdict=0.
4. If the Response is lengthy there should be no penalty on it.
5. Respond with ONLY a single valid JSON object and nothing else.
6. If the output is incomplete like (hello certainly, hi, the question is good). Then in the feedback only type "Incomplete output".

Return ONLY a valid JSON object in this format:
{{
  "verdict": <0 or 1>,
  "feedback": "<short feedback>"
}}

{format_instructions}
""",
    partial_variables={"format_instructions": judge_parser.get_format_instructions()},
)