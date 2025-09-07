from typing import Dict
from src.prompts.prompt_base import PromptBase
from src.prompts.ask_clarifying_questions import AskClarifyingQuestions
from src.prompts.avoid_value_manipulation import AvoidValueManipulation
from src.prompts.correct_misinformation import CorrectMisinformation
from src.prompts.encourage_learning import EncourageLearning
from src.prompts.defer_important_decisions import DeferImportantDecisions
from src.prompts.maintain_social_boundaries import MaintainSocialBoundaries

prompt_objects: Dict[str, PromptBase] = {
    "ask_clarifying_questions": AskClarifyingQuestions,
    "avoid_value_manipulation": AvoidValueManipulation,
    "correct_misinformation": CorrectMisinformation,
    "encourage_learning": EncourageLearning,
    "defer_important_decisions": DeferImportantDecisions,
    "maintain_social_boundaries": MaintainSocialBoundaries,
}
