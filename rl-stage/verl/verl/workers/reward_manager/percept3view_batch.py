from verl.workers.reward_manager import register
from verl.workers.reward_manager.batch import BatchRewardManager


@register("percept3view_batch")
class Percept3ViewBatchRewardManager(BatchRewardManager):
    """Batch reward manager for percept-3view (alias of BatchRewardManager)."""

    pass

