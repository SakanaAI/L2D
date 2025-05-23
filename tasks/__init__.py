from .templates import template_type2template
from .constants import *
from .arc import (
    ARCDataProcessor,
    ARCEvalProcessor,
)
from .gsm8k import (
    GSM8kDataProcessor,
    GSM8kEvalProcessor,
)
from .math import (
    MATHDataProcessor,
    MATHEvalProcessor,
)
from .humaneval import (
    HumanEvalDataProcessor,
    InstructHumanEvalDataProcessor,
)
from .mbpp import (
    MBPPDataProcessor,
)
from .mmlu import (
    MMLUDataProcessor,
    MMLUEvalProcessor,
)
from .mmlu_pro import (
    MMLUProDataProcessor,
    MMLUProEvalProcessor,
)
from .piqa import (
    PIQADataProcessor,
    PIQAEvalProcessor,
)
from .smoltalk import SmolTalkDataProcessor

from .constants import (
    SharedTask2Category,
    SharedTask2Id,
    SharedTasks,
)
